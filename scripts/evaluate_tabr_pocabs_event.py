from __future__ import annotations

import gc
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.special import expit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
TABR_ROOT = ROOT / 'third_party' / 'tabular-dl-tabr-official'
os.environ['PROJECT_DIR'] = str(TABR_ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TABR_ROOT) not in sys.path:
    sys.path.append(str(TABR_ROOT))

import lib
from bin.tabr import Model

sns.set_theme(style='whitegrid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXP_ROOT = TABR_ROOT / 'exp' / 'tabr'
DATA_ROOT = TABR_ROOT / 'data'
OUT_ROOT = ROOT / 'output' / 'apex' / 'datatest' / 'pocabs_event_eval_v2'
CM_ROOT = OUT_ROOT / 'confusion_matrices'
CURVE_ROOT = OUT_ROOT / 'training_curves'
for d in [OUT_ROOT, CM_ROOT / 'event_level', CM_ROOT / 'subject_level', CURVE_ROOT]:
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ['anxiety_rendah', 'anxiety_tinggi']
LABEL_MAP = {'anxiety_rendah': 0, 'anxiety_tinggi': 1}
SUBJECT_METHODS = ['mean', 'top3_mean', 'p90']
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
TOP_K = 3
EXPERIMENTS = [
    {'label': 'qholdout_q123_q4_q5', 'exp_name': 'convat_apex_pocabs_event_qholdout_q123_q4_q5_v2', 'mode': 'qholdout'},
    {'label': 'qwalk_q12_q3_q4', 'exp_name': 'convat_apex_pocabs_event_qwalk_q12_q3_q4_v2', 'mode': 'qwalkforward'},
    {'label': 'qwalk_q123_q4_q5', 'exp_name': 'convat_apex_pocabs_event_qwalk_q123_q4_q5_v2', 'mode': 'qwalkforward'},
]
FEATURE_PATHS = {
    '15_only': [ROOT / 'output/apex/datatest/features/15_04_2026/poc_abs_test.xlsx'],
    '17_only': [ROOT / 'output/apex/datatest/features/17_04_2026/poc_abs_test.xlsx'],
    '15_17_mixed': [
        ROOT / 'output/apex/datatest/features/15_04_2026/poc_abs_test.xlsx',
        ROOT / 'output/apex/datatest/features/17_04_2026/poc_abs_test.xlsx',
    ],
}


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def label_to_int(label: str) -> int:
    return LABEL_MAP[str(label)]


def aggregate_probs(values, method: str, top_k: int = TOP_K) -> float:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) == 0:
        return np.nan
    if method == 'mean':
        return float(np.mean(arr))
    if method == 'p90':
        return float(np.quantile(arr, 0.90))
    if method == 'top3_mean':
        k = min(top_k, len(arr))
        return float(np.mean(np.sort(arr)[-k:]))
    if method == 'direct':
        return float(np.mean(arr))
    raise ValueError(method)


def load_best_seed_report(exp_name: str, seed: int | None = None) -> tuple[int, dict, Path]:
    eval_dir = EXP_ROOT / exp_name / '0-evaluation'
    if seed is not None:
        run_dir = eval_dir / str(seed)
        return seed, json.loads((run_dir / 'report.json').read_text()), run_dir
    rows = []
    for run_dir in sorted([x for x in eval_dir.iterdir() if x.is_dir() and x.name.isdigit()]):
        report_path = run_dir / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            rows.append((int(run_dir.name), report, run_dir))
    if not rows:
        raise FileNotFoundError(f'No evaluation reports found for {exp_name}')
    return max(rows, key=lambda x: x[1]['metrics']['val']['score'])


def load_run_config(exp_name: str, seed: int, report: dict) -> dict:
    seed_toml = EXP_ROOT / exp_name / f'{seed}.toml'
    if seed_toml.exists():
        return lib.load_config(seed_toml)
    if 'config' in report:
        return report['config']
    tuning_toml = EXP_ROOT / exp_name / '0-tuning.toml'
    return lib.load_config(tuning_toml)


def resolve_data_dir(data_path: str) -> Path:
    return DATA_ROOT / data_path.removeprefix(':data/') if data_path.startswith(':data/') else Path(data_path)


def load_model_bundle(exp_name: str, seed: int | None = None) -> dict:
    best_seed, report, run_dir = load_best_seed_report(exp_name, seed=seed)
    cfg = load_run_config(exp_name, best_seed, report)
    data_dir = resolve_data_dir(cfg['data']['path'])
    feature_cols = json.loads((data_dir / 'feature_cols.json').read_text())
    dataset = lib.build_dataset(**cfg['data']).to_torch(device)
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **cfg['model'],
    ).to(device)
    checkpoint = lib.load_checkpoint(run_dir)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return {
        'model': model,
        'dataset': dataset,
        'candidate_x': {key[2:]: dataset.data[key]['train'] for key in dataset.data if key.startswith('X_')},
        'candidate_y': dataset.Y['train'],
        'feature_cols': feature_cols,
        'context_size': cfg['context_size'],
        'seed': best_seed,
        'run_dir': run_dir,
        'data_dir': data_dir,
        'report': report,
    }


def release_bundle(bundle: dict) -> None:
    del bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.inference_mode()
def predict_event_probs(bundle: dict, df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
    feature_cols = bundle['feature_cols']
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        work = pd.concat(
            [df, pd.DataFrame(np.nan, index=df.index, columns=missing_cols)],
            axis=1,
        ).copy()
    else:
        work = df.copy()
    x_np = work.loc[:, feature_cols].to_numpy(dtype=np.float32)
    probs = []
    for start in range(0, len(x_np), batch_size):
        xb = torch.as_tensor(x_np[start:start + batch_size], device=device)
        for i in range(xb.shape[0]):
            logit = bundle['model'](
                x_={'num': xb[i:i + 1]},
                y=None,
                candidate_x_=bundle['candidate_x'],
                candidate_y=bundle['candidate_y'],
                context_size=bundle['context_size'],
                is_train=False,
            ).squeeze(-1).item()
            probs.append(float(expit(logit)))
    return np.asarray(probs, dtype=float)


@torch.inference_mode()
def predict_event_probs_concurrent(bundle: dict, dfs: dict[str, pd.DataFrame], max_workers: int = 3) -> dict[str, np.ndarray]:
    if device.type != 'cpu' or max_workers <= 1 or len(dfs) <= 1:
        return {name: predict_event_probs(bundle, df) for name, df in dfs.items()}
    torch.set_num_threads(1)
    workers = min(max_workers, len(dfs))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {name: executor.submit(predict_event_probs, bundle, df) for name, df in dfs.items()}
        return {name: future.result() for name, future in futures.items()}


def make_event_predictions(frame_df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
    rows = []
    meta_cols = ['date', 'phase', 'participant_raw', 'participant', 'subject_key', 'label', 'true_label', 'question', 'sample', 'event_clip', 'event_no']
    for event_key, g in frame_df.groupby('event_key', sort=False):
        first = g.iloc[0]
        prob = aggregate_probs(g['prob_anxiety_tinggi'], method)
        row = {col: first[col] for col in meta_cols if col in g.columns}
        row.update({
            'event_key': event_key,
            'aggregation': method,
            'threshold': float(threshold),
            'agg_prob_anxiety_tinggi': prob,
            'pred_label': int(prob >= threshold),
            'n_frames': len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def make_subject_predictions(event_df: pd.DataFrame, method: str, threshold: float) -> pd.DataFrame:
    rows = []
    meta_cols = ['date', 'participant_raw', 'participant', 'subject_key', 'label', 'true_label']
    for subject_key, g in event_df.groupby('subject_key', sort=False):
        first = g.iloc[0]
        prob = aggregate_probs(g['agg_prob_anxiety_tinggi'], method)
        row = {col: first[col] for col in meta_cols if col in g.columns}
        row.update({
            'subject_key': subject_key,
            'aggregation': method,
            'threshold': float(threshold),
            'agg_prob_anxiety_tinggi': prob,
            'pred_label': int(prob >= threshold),
            'n_events': len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_predictions(df: pd.DataFrame, level: str, scenario: str, spec: dict, method: str) -> dict:
    y_true = df['true_label'].astype(int).to_numpy()
    y_pred = df['pred_label'].astype(int).to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES, output_dict=True, zero_division=0)
    return {
        'level': level, 'scenario': scenario, 'exp_name': spec['exp_name'], 'label': spec['label'], 'mode': spec['mode'],
        'seed': spec.get('seed'), 'aggregation': method, 'threshold': float(df['threshold'].iloc[0]) if 'threshold' in df.columns and len(df) else np.nan,
        'n': len(df), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'rendah_precision': report['anxiety_rendah']['precision'], 'rendah_recall': report['anxiety_rendah']['recall'], 'rendah_f1': report['anxiety_rendah']['f1-score'],
        'tinggi_precision': report['anxiety_tinggi']['precision'], 'tinggi_recall': report['anxiety_tinggi']['recall'], 'tinggi_f1': report['anxiety_tinggi']['f1-score'],
        'macro_f1': report['macro avg']['f1-score'],
        'recall_gap': abs(report['anxiety_rendah']['recall'] - report['anxiety_tinggi']['recall']),
    }


def plot_confusion(df: pd.DataFrame, level: str, scenario: str, spec: dict, method: str) -> Path:
    cm = confusion_matrix(df['true_label'].astype(int), df['pred_label'].astype(int), labels=[0, 1])
    threshold = float(df['threshold'].iloc[0])
    path = CM_ROOT / level / f"{safe_name(scenario)}__{safe_name(spec['exp_name'])}__seed_{spec.get('seed')}__{method}__thr_{threshold:.3f}.png"
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['rendah', 'tinggi'], yticklabels=['rendah', 'tinggi'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f"{level} | {scenario}\n{spec['label']} | seed={spec.get('seed')} | {method} | thr={threshold:.3f}")
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_training_curves(run_dir: Path, spec: dict) -> Path:
    checkpoint = lib.load_checkpoint(run_dir, map_location='cpu')
    rows = []
    for epoch, item in enumerate(checkpoint['training_log']):
        metrics = item.get('metrics', {})
        epoch_losses = item.get('epoch-losses', [])
        train_loss = metrics.get('train', {}).get('cross-entropy', np.nan)
        if np.isnan(train_loss) and epoch_losses:
            if isinstance(epoch_losses, dict):
                losses = [np.mean(v) for v in epoch_losses.values() if len(v)]
            else:
                losses = epoch_losses
            train_loss = float(np.mean(losses))
        rows.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': metrics.get('val', {}).get('cross-entropy', np.nan),
            'test_loss': metrics.get('test', {}).get('cross-entropy', np.nan),
            'train_acc': metrics.get('train', {}).get('score', np.nan),
            'val_acc': metrics.get('val', {}).get('score', np.nan),
            'test_acc': metrics.get('test', {}).get('score', np.nan),
        })
    df = pd.DataFrame(rows)
    path = CURVE_ROOT / f"{safe_name(spec['exp_name'])}__seed_{spec.get('seed')}.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for col in ['train_loss', 'val_loss', 'test_loss']:
        if col in df and df[col].notna().any():
            axes[0].plot(df['epoch'], df[col], label=col)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    for col in ['train_acc', 'val_acc', 'test_acc']:
        if col in df and df[col].notna().any():
            axes[1].plot(df['epoch'], df[col], label=col)
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    plt.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_datasets_for_calibration(bundle: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for split in ['train', 'val', 'test']:
        df = pd.read_csv(bundle['data_dir'] / f'{split}_split.csv')
        df['true_label'] = df['target'].astype(int)
        df['event_key'] = df['event_id'].astype(str)
        df['subject_key'] = df['participant'].astype(str)
        out[split] = df
    return out


def make_direct_event_predictions(event_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = event_df.copy()
    out['aggregation'] = 'direct'
    out['threshold'] = float(threshold)
    out['agg_prob_anxiety_tinggi'] = out['prob_anxiety_tinggi']
    out['pred_label'] = (out['agg_prob_anxiety_tinggi'] >= threshold).astype(int)
    return out


def calibrate_on_val(bundle: dict, spec: dict) -> tuple[dict, pd.DataFrame]:
    split_df = load_datasets_for_calibration(bundle)['val'].copy()
    split_df['prob_anxiety_tinggi'] = predict_event_probs(bundle, split_df)
    candidates = []
    for threshold in THRESHOLDS:
        event_df = make_direct_event_predictions(split_df, float(threshold))
        summary = summarize_predictions(event_df, 'event_level_val', 'internal_val', spec, 'direct')
        candidates.append((summary, event_df))
    best_summary, best_event_df = max(candidates, key=lambda x: (x[0]['balanced_accuracy'], x[0]['macro_f1'], -x[0]['recall_gap'], x[0]['accuracy']))
    return best_summary, best_event_df


def flatten_frames_to_events(df: pd.DataFrame, base_feature_cols: list[str], event_col: str = 'event_key') -> pd.DataFrame:
    sort_col = 'frame' if 'frame' in df.columns else None
    meta_cols = [c for c in df.columns if c not in base_feature_cols]
    rows = []
    max_len = 0
    for event_key, g in df.groupby(event_col, sort=False):
        if sort_col:
            g = g.sort_values(sort_col, kind='mergesort')
        row = {col: g[col].iloc[0] for col in meta_cols}
        row[event_col] = event_key
        row['n_event_frames'] = int(len(g))
        row['start_frame'] = g[sort_col].iloc[0] if sort_col else np.nan
        row['end_frame'] = g[sort_col].iloc[-1] if sort_col else np.nan
        values = g[base_feature_cols].reset_index(drop=True)
        max_len = max(max_len, len(values))
        for i, (_, frame_row) in enumerate(values.iterrows(), start=1):
            suffix = f'_t{i:03d}'
            for feature in base_feature_cols:
                row[f'{feature}{suffix}'] = frame_row[feature]
        rows.append(row)
    event_df = pd.DataFrame(rows)
    event_feature_cols = [f'{feature}_t{i:03d}' for i in range(1, max_len + 1) for feature in base_feature_cols]
    for col in event_feature_cols:
        if col not in event_df.columns:
            event_df[col] = np.nan
    group_cols = [c for c in ['date', 'participant_raw', 'question', 'sample', 'event_clip'] if c in event_df.columns]
    if group_cols:
        event_df = event_df.sort_values(group_cols + ['start_frame', 'end_frame']).copy()
        event_df['prev_event_end_frame'] = event_df.groupby(group_cols, dropna=False)['end_frame'].shift(1)
        event_df['gap_from_prev_event_frames'] = event_df['start_frame'] - event_df['prev_event_end_frame']
        event_df = event_df.sort_index()
    return event_df


def load_datatest(paths: list[Path], feature_cols: list[str] | None = None) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_excel(path)
        if 'date' not in df.columns:
            df['date'] = path.parent.name
        dfs.append(df)
    frame_df = pd.concat(dfs, ignore_index=True)
    frame_df = frame_df[frame_df['label'].isin(LABEL_MAP)].copy()
    frame_df['true_label'] = frame_df['label'].map(label_to_int).astype(int)
    frame_df['target'] = frame_df['true_label']
    frame_df['event_key'] = (
        frame_df['date'].astype(str) + '|' + frame_df['participant_raw'].astype(str) + '|' + frame_df['question'].astype(str) + '|' +
        frame_df['sample'].astype(str) + '|' + frame_df['event_clip'].astype(str) + '|' + frame_df['event_no'].astype(str)
    )
    frame_df['event_id'] = frame_df['event_key']
    frame_df['subject_key'] = frame_df['date'].astype(str) + '|' + frame_df['participant_raw'].astype(str)
    meta_cols = {
        'date', 'phase', 'condition', 'label', 'participant', 'participant_raw', 'question', 'question_no',
        'sample', 'clip', 'event_clip', 'event_no', 'clip_path', 'frame', 'score', 'annotation_sheet',
        'target', 'true_label', 'event_key', 'event_id', 'subject_key',
    }
    if feature_cols:
        base_feature_cols = sorted({c.rsplit('_t', 1)[0] for c in feature_cols if '_t' in c})
        base_feature_cols = [c for c in base_feature_cols if c in frame_df.columns]
    else:
        base_feature_cols = [c for c in frame_df.columns if c not in meta_cols]
    event_df = flatten_frames_to_events(frame_df, base_feature_cols, event_col='event_key')
    if 'event_id' not in event_df.columns:
        event_df['event_id'] = event_df['event_key']
    return event_df


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    excel_max_cols = 16_384
    with pd.ExcelWriter(OUT_ROOT / 'pocabs_event_external_results.xlsx') as writer:
        for name, df in outputs.items():
            excel_df = df.iloc[:, :excel_max_cols] if df.shape[1] > excel_max_cols else df
            excel_df.to_excel(writer, sheet_name=name[:31], index=False)
    for name, df in outputs.items():
        df.to_csv(OUT_ROOT / f'{name}.csv', index=False)


def write_detail_csv(df: pd.DataFrame, kind: str, spec: dict, scenario: str) -> Path:
    path = OUT_ROOT / f"{kind}__{safe_name(spec['exp_name'])}__seed_{spec.get('seed')}__{safe_name(scenario)}.csv"
    df.to_csv(path, index=False)
    return path


def evaluate_all(return_predictions: bool = False) -> dict[str, pd.DataFrame]:
    calibration_rows = []
    event_rows = []
    subject_rows = []
    event_pred_parts = []
    subject_pred_parts = []
    fp_fn_parts = []
    detail_file_rows = []

    for base_spec in EXPERIMENTS:
        exp_eval_dir = EXP_ROOT / base_spec['exp_name'] / '0-evaluation'
        seed_dirs = sorted([x for x in exp_eval_dir.iterdir() if x.is_dir() and x.name.isdigit()], key=lambda p: int(p.name))
        best = None
        best_bundle = None
        for seed_dir in seed_dirs:
            seed = int(seed_dir.name)
            spec = {**base_spec, 'seed': seed}
            bundle = load_model_bundle(base_spec['exp_name'], seed=seed)
            try:
                val_summary, _ = calibrate_on_val(bundle, spec)
                calibration_rows.append(val_summary)
                if best is None or (val_summary['balanced_accuracy'], val_summary['macro_f1'], -val_summary['recall_gap'], val_summary['accuracy']) > (best[0]['balanced_accuracy'], best[0]['macro_f1'], -best[0]['recall_gap'], best[0]['accuracy']):
                    if best_bundle is not None:
                        release_bundle(best_bundle)
                    best = (val_summary, spec)
                    best_bundle = bundle
                    bundle = None
            finally:
                if bundle is not None:
                    release_bundle(bundle)

        if best is None or best_bundle is None:
            raise RuntimeError(f'No calibrated seed for {base_spec["exp_name"]}')
        val_summary, spec = best
        plot_training_curves(best_bundle['run_dir'], spec)
        method = val_summary['aggregation']
        threshold = float(val_summary['threshold'])

        try:
            scenario_event_dfs = {
                scenario: load_datatest(paths, feature_cols=best_bundle['feature_cols'])
                for scenario, paths in FEATURE_PATHS.items()
            }
            scenario_probs = predict_event_probs_concurrent(best_bundle, scenario_event_dfs)
            for scenario, event_df in scenario_event_dfs.items():
                event_df['prob_anxiety_tinggi'] = scenario_probs[scenario]
                event_df = make_direct_event_predictions(event_df, threshold)
                event_df['scenario'] = scenario
                event_df['exp_name'] = spec['exp_name']
                event_df['model_label'] = spec['label']
                event_df['seed'] = spec['seed']
                if return_predictions:
                    event_pred_parts.append(event_df)
                else:
                    path = write_detail_csv(event_df, 'event_predictions', spec, scenario)
                    detail_file_rows.append({'kind': 'event_predictions', 'scenario': scenario, 'path': str(path), 'size_mb': path.stat().st_size / 1024 / 1024})
                event_summary = summarize_predictions(event_df, 'event_level', scenario, spec, method)
                event_rows.append(event_summary)
                plot_confusion(event_df, 'event_level', scenario, spec, method)

                bad = event_df[event_df['true_label'] != event_df['pred_label']].copy()
                bad['error_type'] = np.where((bad['true_label'] == 0) & (bad['pred_label'] == 1), 'FP', 'FN')
                if return_predictions:
                    fp_fn_parts.append(bad)
                else:
                    path = write_detail_csv(bad, 'fp_fn_events', spec, scenario)
                    detail_file_rows.append({'kind': 'fp_fn_events', 'scenario': scenario, 'path': str(path), 'size_mb': path.stat().st_size / 1024 / 1024})

                subject_df = make_subject_predictions(event_df, method, threshold)
                subject_df['scenario'] = scenario
                subject_df['exp_name'] = spec['exp_name']
                subject_df['model_label'] = spec['label']
                subject_df['seed'] = spec['seed']
                if return_predictions:
                    subject_pred_parts.append(subject_df)
                else:
                    path = write_detail_csv(subject_df, 'subject_predictions', spec, scenario)
                    detail_file_rows.append({'kind': 'subject_predictions', 'scenario': scenario, 'path': str(path), 'size_mb': path.stat().st_size / 1024 / 1024})
                subject_rows.append(summarize_predictions(subject_df, 'subject_level', scenario, spec, method))
                plot_confusion(subject_df, 'subject_level', scenario, spec, method)
        finally:
            release_bundle(best_bundle)

    outputs = {
        'calibration': pd.DataFrame(calibration_rows),
        'event_summary': pd.DataFrame(event_rows),
        'subject_summary': pd.DataFrame(subject_rows),
    }
    if return_predictions:
        outputs.update({
            'event_predictions': pd.concat(event_pred_parts, ignore_index=True) if event_pred_parts else pd.DataFrame(),
            'subject_predictions': pd.concat(subject_pred_parts, ignore_index=True) if subject_pred_parts else pd.DataFrame(),
            'fp_fn_events': pd.concat(fp_fn_parts, ignore_index=True) if fp_fn_parts else pd.DataFrame(),
        })
    else:
        outputs['detail_files'] = pd.DataFrame(detail_file_rows)
    write_outputs(outputs)
    return outputs


if __name__ == '__main__':
    result = evaluate_all()
    print({k: v.shape for k, v in result.items()})
