from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TABR_ROOT = ROOT / 'third_party' / 'tabular-dl-tabr-official'
os.environ['PROJECT_DIR'] = str(TABR_ROOT)
if str(TABR_ROOT) not in sys.path:
    sys.path.append(str(TABR_ROOT))

import lib

FEATURES_PATH = ROOT / 'output' / 'apex' / 'features' / 'poc_abs_flatten_ordered.xlsx'
BASE_TUNING_TOML = TABR_ROOT / 'exp' / 'tabr' / 'convat_apex_anxiety' / '0-tuning.toml'
DATA_ROOT = TABR_ROOT / 'data'
EXP_ROOT = TABR_ROOT / 'exp' / 'tabr'
VERSION = 'v2'
LABEL_MAP = {'anxiety_rendah': 0, 'anxiety_tinggi': 1}
META_COLS = {
    'date', 'phase', 'condition', 'label', 'participant', 'participant_raw', 'question', 'question_no',
    'sample', 'clip', 'event_clip', 'event_no', 'clip_path', 'frame', 'score', 'annotation_sheet',
    'target', 'event_id', 'n_event_frames', 'start_frame', 'end_frame', 'prev_event_end_frame',
    'gap_from_prev_event_frames',
}
EXPERIMENTS = [
    {
        'mode': 'qholdout',
        'dataset_name': f'convat_apex_pocabs_event_qholdout_q123_q4_q5_{VERSION}',
        'exp_name': f'convat_apex_pocabs_event_qholdout_q123_q4_q5_{VERSION}',
        'train_questions': (1, 2, 3),
        'val_questions': (4,),
        'test_questions': (5,),
    },
    {
        'mode': 'qwalkforward',
        'dataset_name': f'convat_apex_pocabs_event_qwalk_q12_q3_q4_{VERSION}',
        'exp_name': f'convat_apex_pocabs_event_qwalk_q12_q3_q4_{VERSION}',
        'train_questions': (1, 2),
        'val_questions': (3,),
        'test_questions': (4,),
    },
    {
        'mode': 'qwalkforward',
        'dataset_name': f'convat_apex_pocabs_event_qwalk_q123_q4_q5_{VERSION}',
        'exp_name': f'convat_apex_pocabs_event_qwalk_q123_q4_q5_{VERSION}',
        'train_questions': (1, 2, 3),
        'val_questions': (4,),
        'test_questions': (5,),
    },
]


def make_event_id(df: pd.DataFrame) -> pd.Series:
    preferred = ['phase', 'participant', 'question', 'sample', 'clip', 'event_clip', 'event_no']
    fallback = ['phase', 'participant', 'question', 'clip', 'event_clip']
    cols = preferred if all(c in df.columns for c in preferred) else fallback
    return df[cols].astype(str).agg('||'.join, axis=1)


def _first_consistent(g: pd.DataFrame, col: str):
    values = g[col].dropna().unique()
    if len(values) > 1:
        raise ValueError(f'Inconsistent {col} values inside event {g.name}: {values[:10]}')
    return g[col].iloc[0]


def add_event_gap_metadata(event_df: pd.DataFrame) -> pd.DataFrame:
    """Add frame distance from previous spotting/event in the same local clip context."""
    group_cols = [c for c in ['phase', 'participant', 'question', 'sample', 'clip', 'event_clip'] if c in event_df.columns]
    if not group_cols or 'start_frame' not in event_df.columns or 'end_frame' not in event_df.columns:
        event_df['prev_event_end_frame'] = np.nan
        event_df['gap_from_prev_event_frames'] = np.nan
        return event_df
    event_df = event_df.sort_values(group_cols + ['start_frame', 'end_frame']).copy()
    event_df['prev_event_end_frame'] = event_df.groupby(group_cols, dropna=False)['end_frame'].shift(1)
    event_df['gap_from_prev_event_frames'] = event_df['start_frame'] - event_df['prev_event_end_frame']
    return event_df.sort_index()


def flatten_frames_to_events(df: pd.DataFrame, base_feature_cols: list[str], event_col: str = 'event_id') -> tuple[pd.DataFrame, list[str]]:
    """Convert frame-level rows into native event-level rows for TabR.

    Each event becomes one row. Frames are sorted by `frame`; every frame feature is expanded
    into temporal columns such as `feature_t001`, `feature_t002`, ... up to the longest event.
    """
    if event_col not in df.columns:
        raise KeyError(event_col)
    sort_col = 'frame' if 'frame' in df.columns else None
    meta_cols = [c for c in df.columns if c not in base_feature_cols]
    rows = []
    max_len = 0
    for event_id, g in df.groupby(event_col, sort=False):
        if sort_col:
            g = g.sort_values(sort_col, kind='mergesort')
        row = {}
        for col in meta_cols:
            if col in {'label', 'target', 'question_no', 'question'}:
                row[col] = _first_consistent(g, col)
            else:
                row[col] = g[col].iloc[0]
        row[event_col] = event_id
        row['n_event_frames'] = int(len(g))
        if sort_col:
            row['start_frame'] = g[sort_col].iloc[0]
            row['end_frame'] = g[sort_col].iloc[-1]
        else:
            row['start_frame'] = np.nan
            row['end_frame'] = np.nan
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
    event_df = add_event_gap_metadata(event_df)
    ordered_meta = [c for c in event_df.columns if c not in event_feature_cols]
    return event_df[ordered_meta + event_feature_cols], event_feature_cols


def load_base_dataframe() -> tuple[pd.DataFrame, list[str]]:
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(FEATURES_PATH)
    frame_df = pd.read_excel(FEATURES_PATH)
    frame_df = frame_df[frame_df['label'].isin(LABEL_MAP)].copy()
    frame_df['target'] = frame_df['label'].map(LABEL_MAP).astype(np.int64)
    frame_df['event_id'] = make_event_id(frame_df)
    base_feature_cols = [c for c in frame_df.columns if c not in META_COLS]
    return flatten_frames_to_events(frame_df, base_feature_cols, event_col='event_id')


def split_by_questions(df: pd.DataFrame, train_questions: tuple[int, ...], val_questions: tuple[int, ...], test_questions: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_table = df[['event_id', 'question_no']].drop_duplicates().reset_index(drop=True)
    split_ids = []
    for questions in [train_questions, val_questions, test_questions]:
        split_ids.append(set(event_table[event_table['question_no'].isin(questions)]['event_id']))
    if split_ids[0] & split_ids[1] or split_ids[0] & split_ids[2] or split_ids[1] & split_ids[2]:
        raise RuntimeError('Event-id overlap detected between splits')
    return tuple(df[df['event_id'].isin(ids)].copy() for ids in split_ids)


def describe_split(name: str, df: pd.DataFrame) -> dict:
    return {
        'split': name,
        'rows': int(len(df)),
        'events': int(df['event_id'].nunique()) if 'event_id' in df.columns else int(len(df)),
        'subjects': int(df['participant'].nunique()) if 'participant' in df.columns else None,
        'event_labels': df['label'].value_counts().to_dict(),
        'frame_count_min': int(df['n_event_frames'].min()) if 'n_event_frames' in df.columns and len(df) else None,
        'frame_count_max': int(df['n_event_frames'].max()) if 'n_event_frames' in df.columns and len(df) else None,
        'frame_count_mean': float(df['n_event_frames'].mean()) if 'n_event_frames' in df.columns and len(df) else None,
    }


def export_one(cfg: dict, overwrite: bool = False, df: pd.DataFrame | None = None, feature_cols: list[str] | None = None) -> dict:
    if df is None or feature_cols is None:
        df, feature_cols = load_base_dataframe()
    out_dir = DATA_ROOT / cfg['dataset_name']
    exp_dir = EXP_ROOT / cfg['exp_name']
    required = ['X_num_train.npy', 'X_num_val.npy', 'X_num_test.npy', 'Y_train.npy', 'Y_val.npy', 'Y_test.npy', 'train_split.csv', 'val_split.csv', 'test_split.csv', 'feature_cols.json', 'READY']
    existing_complete = out_dir.exists() and all((out_dir / name).exists() for name in required)
    exp_config_exists = (exp_dir / '0-tuning.toml').exists()
    if out_dir.exists() and not overwrite and not existing_complete:
        raise FileExistsError(f'Incomplete dataset already exists: {out_dir}')
    if exp_dir.exists() and not overwrite and not exp_config_exists:
        raise FileExistsError(f'Incomplete experiment already exists: {exp_dir}')

    df_train, df_val, df_test = split_by_questions(df, cfg['train_questions'], cfg['val_questions'], cfg['test_questions'])
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(df_train[feature_cols])).astype(np.float32)
    X_val = scaler.transform(imputer.transform(df_val[feature_cols])).astype(np.float32)
    X_test = scaler.transform(imputer.transform(df_test[feature_cols])).astype(np.float32)

    if existing_complete and exp_config_exists and not overwrite:
        return {
            'dataset_dir': str(out_dir),
            'exp_dir': str(exp_dir),
            'tuning_toml': str(exp_dir / '0-tuning.toml'),
            'feature_count': len(feature_cols),
            'splits': [describe_split('train', df_train), describe_split('val', df_val), describe_split('test', df_test)],
            'status': 'already_exists_complete',
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / 'X_num_train.npy', X_train)
    np.save(out_dir / 'X_num_val.npy', X_val)
    np.save(out_dir / 'X_num_test.npy', X_test)
    np.save(out_dir / 'Y_train.npy', df_train['target'].to_numpy(dtype=np.int64))
    np.save(out_dir / 'Y_val.npy', df_val['target'].to_numpy(dtype=np.int64))
    np.save(out_dir / 'Y_test.npy', df_test['target'].to_numpy(dtype=np.int64))
    (out_dir / 'READY').write_text('')
    (out_dir / 'info.json').write_text(json.dumps({'task_type': 'binclass', 'name': cfg['dataset_name'], 'id': cfg['dataset_name']}, indent=2))
    (out_dir / 'feature_cols.json').write_text(json.dumps(feature_cols, indent=2))
    (out_dir / 'split_config.json').write_text(json.dumps({k: list(v) if isinstance(v, tuple) else v for k, v in cfg.items()}, indent=2))
    df_train.to_csv(out_dir / 'train_split.csv', index=False)
    df_val.to_csv(out_dir / 'val_split.csv', index=False)
    df_test.to_csv(out_dir / 'test_split.csv', index=False)

    config = copy.deepcopy(lib.load_config(BASE_TUNING_TOML))
    config['space']['data']['path'] = f':data/{cfg["dataset_name"]}'
    exp_dir.mkdir(parents=True, exist_ok=True)
    tuning_toml = exp_dir / '0-tuning.toml'
    lib.dump_config(config, tuning_toml)

    return {
        'dataset_dir': str(out_dir),
        'exp_dir': str(exp_dir),
        'tuning_toml': str(tuning_toml),
        'feature_count': len(feature_cols),
        'splits': [describe_split('train', df_train), describe_split('val', df_val), describe_split('test', df_test)],
    }


def export_all(overwrite: bool = False) -> list[dict]:
    df, feature_cols = load_base_dataframe()
    return [export_one(cfg, overwrite=overwrite, df=df, feature_cols=feature_cols) for cfg in EXPERIMENTS]


def main() -> None:
    results = export_all(overwrite=False)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
