from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURES_PATH = ROOT / 'output/apex/features/poc_abs_flatten_ordered.xlsx'
OFFICIAL_ROOT = ROOT / 'third_party/tabular-dl-tabr-official'
LABEL_MAP = {'anxiety_rendah': 0, 'anxiety_tinggi': 1}
META_COLS = [
    'phase', 'condition', 'label', 'participant', 'participant_raw', 'question', 'question_no',
    'sample', 'clip', 'event_clip', 'event_no', 'clip_path', 'frame', 'target', 'event_id',
]
FOLDS = [
    {
        'dataset_name': 'convat_apex_anxiety_qwalk_q12_q3_q4',
        'train_questions': (1, 2),
        'val_questions': (3,),
        'test_questions': (4,),
    },
    {
        'dataset_name': 'convat_apex_anxiety_qwalk_q123_q4_q5',
        'train_questions': (1, 2, 3),
        'val_questions': (4,),
        'test_questions': (5,),
    },
]


def load_base_dataframe() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(FEATURES_PATH)
    df = df[df['label'].isin(LABEL_MAP)].copy()
    df['target'] = df['label'].map(LABEL_MAP).astype(np.int64)
    df['event_id'] = (
        df['phase'].astype(str) + '||' + df['participant'].astype(str) + '||' +
        df['question'].astype(str) + '||' + df['clip'].astype(str) + '||' + df['event_clip'].astype(str)
    )
    feature_cols = [c for c in df.columns if c not in META_COLS]
    return df, feature_cols


def make_split_tables(
    df: pd.DataFrame,
    train_questions: tuple[int, ...],
    val_questions: tuple[int, ...],
    test_questions: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    event_table = (
        df[['event_id', 'question_no']]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    train_event_ids = set(event_table[event_table['question_no'].isin(train_questions)]['event_id'])
    val_event_ids = set(event_table[event_table['question_no'].isin(val_questions)]['event_id'])
    test_event_ids = set(event_table[event_table['question_no'].isin(test_questions)]['event_id'])
    df_train = df[df['event_id'].isin(train_event_ids)].copy()
    df_val = df[df['event_id'].isin(val_event_ids)].copy()
    df_test = df[df['event_id'].isin(test_event_ids)].copy()
    return df_train, df_val, df_test


def export_one_fold(
    df: pd.DataFrame,
    feature_cols: list[str],
    dataset_name: str,
    train_questions: tuple[int, ...],
    val_questions: tuple[int, ...],
    test_questions: tuple[int, ...],
) -> dict:
    out_dir = OFFICIAL_ROOT / 'data' / dataset_name
    if out_dir.exists():
        raise FileExistsError(f'Dataset dir already exists: {out_dir}')

    df_train, df_val, df_test = make_split_tables(df, train_questions, val_questions, test_questions)

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(df_train[feature_cols])).astype(np.float32)
    X_val = scaler.transform(imputer.transform(df_val[feature_cols])).astype(np.float32)
    X_test = scaler.transform(imputer.transform(df_test[feature_cols])).astype(np.float32)
    y_train = df_train['target'].to_numpy(dtype=np.int64)
    y_val = df_val['target'].to_numpy(dtype=np.int64)
    y_test = df_test['target'].to_numpy(dtype=np.int64)

    out_dir.mkdir(parents=True, exist_ok=False)
    np.save(out_dir / 'X_num_train.npy', X_train)
    np.save(out_dir / 'X_num_val.npy', X_val)
    np.save(out_dir / 'X_num_test.npy', X_test)
    np.save(out_dir / 'Y_train.npy', y_train)
    np.save(out_dir / 'Y_val.npy', y_val)
    np.save(out_dir / 'Y_test.npy', y_test)
    (out_dir / 'READY').write_text('')
    (out_dir / 'info.json').write_text(json.dumps({'task_type': 'binclass', 'name': dataset_name, 'id': dataset_name}, indent=2))
    (out_dir / 'feature_cols.json').write_text(json.dumps(feature_cols, indent=2))
    df_train.to_csv(out_dir / 'train_split.csv', index=False)
    df_val.to_csv(out_dir / 'val_split.csv', index=False)
    df_test.to_csv(out_dir / 'test_split.csv', index=False)

    return {
        'dataset_name': dataset_name,
        'dataset_dir': out_dir,
        'train_questions': train_questions,
        'val_questions': val_questions,
        'test_questions': test_questions,
        'df_train': df_train,
        'df_val': df_val,
        'df_test': df_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }


def export_all_folds() -> list[dict]:
    df, feature_cols = load_base_dataframe()
    return [
        export_one_fold(
            df=df,
            feature_cols=feature_cols,
            dataset_name=cfg['dataset_name'],
            train_questions=cfg['train_questions'],
            val_questions=cfg['val_questions'],
            test_questions=cfg['test_questions'],
        )
        for cfg in FOLDS
    ]


def main() -> None:
    results = export_all_folds()
    print(json.dumps([
        {
            'dataset_name': item['dataset_name'],
            'dataset_dir': str(item['dataset_dir']),
            'train_questions': item['train_questions'],
            'val_questions': item['val_questions'],
            'test_questions': item['test_questions'],
        }
        for item in results
    ], indent=2))


if __name__ == '__main__':
    main()
