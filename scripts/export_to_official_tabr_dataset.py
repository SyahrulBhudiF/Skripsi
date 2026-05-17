from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_SEED = 72
N_EXTERNAL_PER_LABEL = 20
ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / 'output/apex/features/poc_abs_flatten_ordered.xlsx'
OFFICIAL_ROOT = ROOT / 'third_party/tabular-dl-tabr-official'
DATASET_NAME = 'convat_apex_anxiety'
OUT_DIR = OFFICIAL_ROOT / 'data' / DATASET_NAME


def balanced_external_events(event_table: pd.DataFrame, n_per_label: int, seed: int = 42) -> set[str]:
    rng = random.Random(seed)
    selected_ids: list[str] = []
    for label_name in ['anxiety_rendah', 'anxiety_tinggi']:
        label_df = event_table[event_table['label'] == label_name].copy()
        picked_rows = []
        used_ids = set()
        participant_groups = []
        for participant, part_df in label_df.groupby('participant', sort=True):
            part_df = part_df.sort_values(['phase', 'question', 'clip', 'event_clip'], kind='stable')
            phase_groups = []
            for phase, phase_df in part_df.groupby('phase', sort=True):
                phase_groups.append(phase_df.to_dict('records'))
            participant_groups.append((participant, phase_groups))
        while len(picked_rows) < n_per_label:
            progress = False
            for _participant, phase_groups in participant_groups:
                for records in phase_groups:
                    while records and records[0]['event_id'] in used_ids:
                        records.pop(0)
                    if not records:
                        continue
                    row = records.pop(0)
                    picked_rows.append(row)
                    used_ids.add(row['event_id'])
                    progress = True
                    if len(picked_rows) >= n_per_label:
                        break
                if len(picked_rows) >= n_per_label:
                    break
            if not progress:
                break
        if len(picked_rows) < n_per_label:
            remaining = label_df[~label_df['event_id'].isin(used_ids)].sort_values(['participant', 'phase', 'question', 'clip', 'event_clip'], kind='stable')
            for row in remaining.to_dict('records'):
                picked_rows.append(row)
                used_ids.add(row['event_id'])
                if len(picked_rows) >= n_per_label:
                    break
        selected_ids.extend([row['event_id'] for row in picked_rows[:n_per_label]])
    return set(selected_ids)


def balance_train_clips_by_label(df_train: pd.DataFrame, seed: int = 42):
    rng = random.Random(seed)
    event_table_train = (
        df_train[['event_id', 'label', 'target', 'phase', 'participant', 'question', 'clip', 'event_clip']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grouped = {}
    for label_name, label_df in event_table_train.groupby('label', sort=True):
        clip_table = (
            label_df[['participant', 'question', 'clip', 'label']]
            .drop_duplicates()
            .sort_values(['participant', 'question', 'clip'], kind='stable')
            .reset_index(drop=True)
        )
        grouped[label_name] = clip_table
    min_clip_count = min(len(x) for x in grouped.values())
    selected_clip_keys = []
    for label_name, clip_table in grouped.items():
        clip_records = clip_table.to_dict('records')
        rng.shuffle(clip_records)
        chosen = clip_records[:min_clip_count]
        selected_clip_keys.extend([
            (row['participant'], row['question'], row['clip'], row['label'])
            for row in chosen
        ])
    selected_clip_keys = set(selected_clip_keys)
    return df_train[
        df_train.apply(lambda row: (row['participant'], row['question'], row['clip'], row['label']) in selected_clip_keys, axis=1)
    ].copy()


def main():
    df = pd.read_excel(FEATURES_PATH)
    label_map = {'anxiety_rendah': 0, 'anxiety_tinggi': 1}
    df = df[df['label'].isin(label_map)].copy()
    df['target'] = df['label'].map(label_map)
    df['event_id'] = (
        df['phase'].astype(str) + '||' + df['participant'].astype(str) + '||' +
        df['question'].astype(str) + '||' + df['clip'].astype(str) + '||' + df['event_clip'].astype(str)
    )

    event_table = df[['event_id', 'label', 'target', 'phase', 'participant', 'question', 'clip', 'event_clip']].drop_duplicates().reset_index(drop=True)
    external_event_ids = balanced_external_events(event_table, N_EXTERNAL_PER_LABEL, seed=RANDOM_SEED)
    df_external = df[df['event_id'].isin(external_event_ids)].copy()
    df_train_all = df[~df['event_id'].isin(external_event_ids)].copy()
    df_train_all_balanced = balance_train_clips_by_label(df_train_all, seed=RANDOM_SEED)
    train_event_table_balanced = df_train_all_balanced[['event_id', 'label', 'target']].drop_duplicates().reset_index(drop=True)

    train_event_ids, val_event_ids = train_test_split(
        train_event_table_balanced['event_id'],
        test_size=0.3,
        stratify=train_event_table_balanced['target'],
        random_state=RANDOM_SEED,
    )
    train_event_ids = set(train_event_ids.tolist())
    val_event_ids = set(val_event_ids.tolist())

    df_tr = df_train_all_balanced[df_train_all_balanced['event_id'].isin(train_event_ids)].copy()
    df_val = df_train_all_balanced[df_train_all_balanced['event_id'].isin(val_event_ids)].copy()

    meta_cols = [
        'phase', 'condition', 'label', 'participant', 'participant_raw', 'question', 'question_no',
        'sample', 'clip', 'event_clip', 'event_no', 'clip_path', 'frame', 'target', 'event_id',
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(df_tr[feature_cols])).astype(np.float32)
    X_val = scaler.transform(imputer.transform(df_val[feature_cols])).astype(np.float32)
    X_test = scaler.transform(imputer.transform(df_external[feature_cols])).astype(np.float32)
    y_train = df_tr['target'].to_numpy(dtype=np.int64)
    y_val = df_val['target'].to_numpy(dtype=np.int64)
    y_test = df_external['target'].to_numpy(dtype=np.int64)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / 'X_num_train.npy', X_train)
    np.save(OUT_DIR / 'X_num_val.npy', X_val)
    np.save(OUT_DIR / 'X_num_test.npy', X_test)
    np.save(OUT_DIR / 'Y_train.npy', y_train)
    np.save(OUT_DIR / 'Y_val.npy', y_val)
    np.save(OUT_DIR / 'Y_test.npy', y_test)
    (OUT_DIR / 'READY').write_text('')
    (OUT_DIR / 'info.json').write_text(json.dumps({'task_type': 'binclass', 'name': 'Convat Apex Anxiety', 'id': DATASET_NAME}, indent=2))
    (OUT_DIR / 'feature_cols.json').write_text(json.dumps(feature_cols, indent=2))
    df_tr.to_csv(OUT_DIR / 'train_split.csv', index=False)
    df_val.to_csv(OUT_DIR / 'val_split.csv', index=False)
    df_external.to_csv(OUT_DIR / 'test_split.csv', index=False)
    print(OUT_DIR)


if __name__ == '__main__':
    main()
