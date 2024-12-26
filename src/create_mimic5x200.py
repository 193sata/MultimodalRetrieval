import pandas as pd
import numpy as np

all_conditions = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']
condition_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

df = pd.read_csv('../data/raw/mimic-cxr-2.0.0-chexpert.csv')

# 排他的陽性のデータのみ抽出
df = df[~(df[all_conditions] == -1.0).any(axis=1)]
df = df[df[all_conditions].sum(axis=1) == 1.0]

split_info = pd.read_csv('../data/raw/mimic-cxr-2.0.0-split.csv')

df_merged = pd.merge(df, split_info, on=['subject_id', 'study_id'])

df = df_merged[['subject_id', 'study_id', 'dicom_id', 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion', 'split']]

# 5つの疾患のいずれか1つだけ陽性のデータのみ抽出
df = df[df[condition_cols].sum(axis=1) == 1.0]

# 欠落値または0.0を0、1.0を1に変換
df[condition_cols] = df[condition_cols].fillna(0)  # 欠落値を0に置換
df[condition_cols] = df[condition_cols].astype(float).astype(int)  # 0.0/1.0を整数の0/1に変換

# テストデータの作成 各クラスに対してクエリが10件、データセットが200件
# できる限りtestとvalidateからサンプリングし、それでも足りない場合はtrainからサンプリング
for i in condition_cols:
    candidates = df[(df['split'] == 'test') & (df[i] == 1.0)]
    validates = df[(df['split'] == 'validate') & (df[i] == 1.0)]
    trains = df[(df['split'] == 'train') & (df[i] == 1.0)]

    query_samples = candidates.sample(n=10)
    remaining_candidates = candidates.drop(query_samples.index)

    if len(remaining_candidates) < 200:
        needed = 200 - len(remaining_candidates)
        
        # validateから追加
        if len(validates) >= needed:
            additional_samples = validates.sample(n=needed)
            remaining_candidates = pd.concat([remaining_candidates, additional_samples])
        else:
            remaining_candidates = pd.concat([remaining_candidates, validates])
            
            still_needed = 200 - len(remaining_candidates)
            if still_needed > 0:
                if len(trains) >= still_needed:
                    additional_train_samples = trains.sample(n=still_needed)
                    remaining_candidates = pd.concat([remaining_candidates, additional_train_samples])
                else:
                    remaining_candidates = pd.concat([remaining_candidates, trains])
    
    dataset_samples = remaining_candidates.sample(n=200)

    df.loc[query_samples.index, 'split'] = 'query'
    df.loc[dataset_samples.index, 'split'] = 'dataset'

# トレーニングデータとバリデーションデータを作成 9:1に分割
remaining_data = df[~df['split'].isin(['query', 'dataset'])]

train_size = int(len(remaining_data) * 0.9)

train_indices = np.random.choice(remaining_data.index, size=train_size, replace=False)

df.loc[train_indices, 'split'] = 'train'
df.loc[~df.index.isin(train_indices) & ~df['split'].isin(['query', 'dataset']), 'split'] = 'validate'

# df.to_csv('../../data/mimic_data.csv', index=False)

# ファイルごとに保存
split_files = {
    'train': '../data/train.csv',
    'validate': '../data/validate.csv',
    'query': '../data/query.csv',
    'dataset': '../data/candidate.csv'
}
for split, file_path in split_files.items():
    split_df = df[df['split'] == split].drop(columns=['split'])
    split_df.to_csv(file_path, index=False)