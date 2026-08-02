from __future__ import annotations

import gc
import json
import os
import re
import sys
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
OUT_ROOT = ROOT / 'output' / 'apex' / 'datatest' / 'pocabs_bias_v1'
CM_ROOT = OUT_ROOT / 'confusion_matrices'
for d in [OUT_ROOT, CM_ROOT / 'frame_level', CM_ROOT / 'event_level', CM_ROOT / 'subject_level']:
    d.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ['anxiety_rendah', 'anxiety_tinggi']
LABEL_MAP = {'anxiety_rendah': 0, 'anxiety_tinggi': 1}
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.025), 3)
TOP_K = 3
SUBJECT_METHODS = ['mean', 'median', 'trimmed_mean', 'p90', 'top3_mean', 'proportion', 'count']
SUBJECT_EVENT_THRESHOLDS = np.round(np.arange(0.20, 0.951, 0.025), 3)
EXPERIMENTS = [
    {'label': 'qholdout_q123_q4_q5_alis', 'exp_name': 'convat_apex_anxiety_qholdout_q123_q4_q5_alis', 'mode': 'qholdout'},
    {'label': 'qwalk_q12_q3_q4_alis', 'exp_name': 'convat_apex_anxiety_qwalk_q12_q3_q4_alis', 'mode': 'qwalkforward'},
    {'label': 'qwalk_q123_q4_q5_alis', 'exp_name': 'convat_apex_anxiety_qwalk_q123_q4_q5_alis', 'mode': 'qwalkforward'},
]
FEATURE_PATHS = {
    '15_only': [ROOT / 'output/apex/datatest/features/15_04_2026/poc_abs_test.xlsx'],
    '17_only': [ROOT / 'output/apex/datatest/features/17_04_2026/poc_abs_test.xlsx'],
    '15_17_mixed': [
        ROOT / 'output/apex/datatest/features/15_04_2026/poc_abs_test.xlsx',
        ROOT / 'output/apex/datatest/features/17_04_2026/poc_abs_test.xlsx',
    ],
}
META_COLS = {
    'date', 'phase', 'condition', 'label', 'participant', 'participant_raw', 'question', 'question_no',
    'sample', 'clip', 'event_clip', 'event_no', 'clip_path', 'frame', 'score', 'annotation_sheet',
    'target', 'true_label', 'event_key', 'event_id', 'subject_key',
}


def safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(value))


def aggregate_probs(values, method: str, top_k: int = TOP_K) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    if method == 'mean':
        return float(np.mean(arr))
    if method == 'median':
        return float(np.median(arr))
    if method == 'trimmed_mean':
        if len(arr) < 5:
            return float(np.mean(arr))
        lo, hi = np.quantile(arr, [0.10, 0.90])
        trimmed = arr[(arr >= lo) & (arr <= hi)]
        return float(np.mean(trimmed if len(trimmed) else arr))
    if method == 'p90':
        return float(np.quantile(arr, 0.90))
    if method == 'top3_mean':
        k = min(top_k, len(arr))
        return float(np.mean(np.sort(arr)[-k:]))
    raise ValueError(method)


def aggregate_subject_events(values, method: str, event_threshold: float, top_k: int = TOP_K) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    if method == 'proportion':
        return float(np.mean(arr >= event_threshold))
    if method == 'count':
        return float(np.sum(arr >= event_threshold))
    return aggregate_probs(arr, method, top_k=top_k)


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
    return lib.load_config(EXP_ROOT / exp_name / '0-tuning.toml')


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
def predict_probs(bundle: dict, df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
    feature_cols = bundle['feature_cols']
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        work = pd.concat([df, pd.DataFrame(0.0, index=df.index, columns=missing_cols)], axis=1).copy()
    else:
        work = df.copy()
    x_np = work.loc[:, feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    probs = []
    for start in range(0, len(x_np), batch_size):
        xb = torch.as_tensor(x_np[start:start + batch_size], device=device)
        logits = bundle['model'](
            x_={'num': xb},
            y=None,
            candidate_x_=bundle['candidate_x'],
            candidate_y=bundle['candidate_y'],
            context_size=bundle['context_size'],
            is_train=False,
        ).squeeze(-1).detach().cpu().numpy()
        probs.extend(expit(logits).astype(float).tolist())
    probs_arr = np.asarray(probs, dtype=float)
    if not np.isfinite(probs_arr).all():
        raise RuntimeError('Non-finite frame probabilities')
    return probs_arr


def load_datatest_frames(paths: list[Path]) -> pd.DataFrame:
    dfs = []
    for path in paths:
        df = pd.read_excel(path)
        if 'date' not in df.columns:
            df['date'] = path.parent.name
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df['label'].isin(LABEL_MAP)].copy()
    df['true_label'] = df['label'].map(LABEL_MAP).astype(int)
    df['target'] = df['true_label']
    df['event_key'] = (
        df['date'].astype(str) + '|' + df['participant_raw'].astype(str) + '|' + df['question'].astype(str) + '|' +
        df['sample'].astype(str) + '|' + df['event_clip'].astype(str) + '|' + df['event_no'].astype(str)
    )
    df['event_id'] = df['event_key']
    df['subject_key'] = df['date'].astype(str) + '|' + df['participant_raw'].astype(str)
    return df


def make_group_predictions(df: pd.DataFrame, group_col: str, prob_col: str, method: str, threshold: float) -> pd.DataFrame:
    rows = []
    meta_cols = ['date', 'phase', 'participant_raw', 'participant', 'subject_key', 'label', 'true_label', 'question', 'sample', 'event_clip', 'event_no']
    for key, g in df.groupby(group_col, sort=False):
        first = g.iloc[0]
        prob = aggregate_probs(g[prob_col], method)
        row = {col: first[col] for col in meta_cols if col in g.columns}
        row.update({
            group_col: key,
            'aggregation': method,
            'threshold': float(threshold),
            'agg_prob_anxiety_tinggi': prob,
            'pred_label': int(prob >= threshold),
            'n': len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def make_subject_predictions(
    event_df: pd.DataFrame,
    method: str,
    subject_threshold: float,
    event_threshold: float,
) -> pd.DataFrame:
    rows = []
    meta_cols = ['date', 'participant_raw', 'participant', 'subject_key', 'label', 'true_label']
    for subject_key, g in event_df.groupby('subject_key', sort=False):
        first = g.iloc[0]
        prob = aggregate_subject_events(g['agg_prob_anxiety_tinggi'], method, event_threshold)
        row = {col: first[col] for col in meta_cols if col in g.columns}
        row.update({
            'subject_key': subject_key,
            'aggregation': method,
            'event_threshold': float(event_threshold),
            'threshold': float(subject_threshold),
            'agg_prob_anxiety_tinggi': prob,
            'pred_label': int(prob >= subject_threshold),
            'n_events': len(g),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate_subject_threshold(
    event_df: pd.DataFrame,
    spec: dict,
    scenario: str,
    event_method: str,
    event_threshold: float,
) -> tuple[dict, pd.DataFrame, str, float]:
    candidates = []
    for subject_method in SUBJECT_METHODS:
        thresholds = np.arange(1, max(2, int(event_df.groupby('subject_key').size().max()) + 1)) if subject_method == 'count' else SUBJECT_EVENT_THRESHOLDS
        for subject_threshold in thresholds:
            subject_df = make_subject_predictions(
                event_df,
                subject_method,
                float(subject_threshold),
                event_threshold,
            )
            summary = summarize_predictions(
                subject_df,
                'subject_level',
                scenario,
                spec,
                f'{event_method}->{subject_method}',
            )
            summary['event_threshold'] = float(event_threshold)
            candidates.append((summary, subject_df, subject_method, float(subject_threshold)))
    return max(
        candidates,
        key=lambda x: (
            x[0]['balanced_accuracy'],
            x[0]['macro_f1'],
            -x[0]['recall_gap'],
            x[0]['accuracy'],
        ),
    )


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


def calibrate_on_val(bundle: dict, spec: dict) -> tuple[dict, str, float]:
    val_df = pd.read_csv(bundle['data_dir'] / 'val_split.csv')
    val_df['true_label'] = val_df['target'].astype(int)
    val_df['event_key'] = val_df['event_id'].astype(str)
    val_df['subject_key'] = val_df['participant'].astype(str)
    val_df['prob_anxiety_tinggi'] = predict_probs(bundle, val_df)
    candidates = []
    for method in ['mean', 'top3_mean', 'p90']:
        for threshold in THRESHOLDS:
            event_df = make_group_predictions(val_df, 'event_key', 'prob_anxiety_tinggi', method, float(threshold))
            summary = summarize_predictions(event_df, 'event_level_val', 'internal_val', spec, method)
            candidates.append((summary, method, float(threshold)))
    return max(candidates, key=lambda x: (x[0]['balanced_accuracy'], x[0]['macro_f1'], -x[0]['recall_gap'], x[0]['accuracy']))


def write_outputs(outputs: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(OUT_ROOT / 'results.xlsx') as writer:
        for name, df in outputs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    for name, df in outputs.items():
        df.to_csv(OUT_ROOT / f'{name}.csv', index=False)


def evaluate_all(return_predictions: bool = False) -> dict[str, pd.DataFrame]:
    calibration_rows = []
    frame_rows = []
    event_rows = []
    subject_rows = []
    detail_rows = []
    frame_pred_parts = []
    event_pred_parts = []
    subject_pred_parts = []

    for base_spec in EXPERIMENTS:
        seed, _, _ = load_best_seed_report(base_spec['exp_name'])
        spec = {**base_spec, 'seed': seed}
        bundle = load_model_bundle(base_spec['exp_name'], seed=seed)
        try:
            val_summary, method, threshold = calibrate_on_val(bundle, spec)
            calibration_rows.append(val_summary)
            for scenario, paths in FEATURE_PATHS.items():
                frame_df = load_datatest_frames(paths)
                frame_df['prob_anxiety_tinggi'] = predict_probs(bundle, frame_df)
                frame_df['aggregation'] = 'frame_direct'
                frame_df['threshold'] = threshold
                frame_df['pred_label'] = (frame_df['prob_anxiety_tinggi'] >= threshold).astype(int)
                frame_df['scenario'] = scenario
                frame_df['exp_name'] = spec['exp_name']
                frame_df['model_label'] = spec['label']
                frame_df['seed'] = spec['seed']
                frame_rows.append(summarize_predictions(frame_df, 'frame_level', scenario, spec, 'frame_direct'))
                plot_confusion(frame_df, 'frame_level', scenario, spec, 'frame_direct')

                event_df = make_group_predictions(frame_df, 'event_key', 'prob_anxiety_tinggi', method, threshold)
                event_df['scenario'] = scenario
                event_df['exp_name'] = spec['exp_name']
                event_df['model_label'] = spec['label']
                event_df['seed'] = spec['seed']
                event_rows.append(summarize_predictions(event_df, 'event_level', scenario, spec, method))
                plot_confusion(event_df, 'event_level', scenario, spec, method)

                subject_summary, subject_df, subject_method, subject_threshold = calibrate_subject_threshold(
                    event_df,
                    spec,
                    scenario,
                    method,
                    threshold,
                )
                subject_df['scenario'] = scenario
                subject_df['exp_name'] = spec['exp_name']
                subject_df['model_label'] = spec['label']
                subject_df['seed'] = spec['seed']
                subject_df['event_aggregation'] = method
                subject_df['subject_aggregation'] = subject_method
                subject_rows.append(subject_summary)
                plot_confusion(subject_df, 'subject_level', scenario, spec, f'{method}_to_{subject_method}')

                if return_predictions:
                    frame_pred_parts.append(frame_df)
                    event_pred_parts.append(event_df)
                    subject_pred_parts.append(subject_df)
                else:
                    for kind, df in [('frame_predictions', frame_df), ('event_predictions', event_df), ('subject_predictions', subject_df)]:
                        path = OUT_ROOT / f"{kind}__{safe_name(spec['exp_name'])}__seed_{spec['seed']}__{safe_name(scenario)}.csv"
                        df.to_csv(path, index=False)
                        detail_rows.append({'kind': kind, 'scenario': scenario, 'path': str(path), 'size_mb': path.stat().st_size / 1024 / 1024})
        finally:
            release_bundle(bundle)

    outputs = {
        'calibration': pd.DataFrame(calibration_rows),
        'frame_summary': pd.DataFrame(frame_rows),
        'event_summary': pd.DataFrame(event_rows),
        'subject_summary': pd.DataFrame(subject_rows),
    }
    if return_predictions:
        outputs.update({
            'frame_predictions': pd.concat(frame_pred_parts, ignore_index=True) if frame_pred_parts else pd.DataFrame(),
            'event_predictions': pd.concat(event_pred_parts, ignore_index=True) if event_pred_parts else pd.DataFrame(),
            'subject_predictions': pd.concat(subject_pred_parts, ignore_index=True) if subject_pred_parts else pd.DataFrame(),
        })
    else:
        outputs['detail_files'] = pd.DataFrame(detail_rows)
    write_outputs(outputs)
    return outputs


if __name__ == '__main__':
    result = evaluate_all(return_predictions=False)
    print({k: v.shape for k, v in result.items()})
