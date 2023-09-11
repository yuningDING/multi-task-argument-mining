import pandas as pd
import numpy as np


def get_slice(name, dataframe):
    id_list = pd.read_csv(name + '_ids.csv', encoding='utf-8')['discourse_id'].to_numpy()
    print(id_list)
    dataframe = dataframe[dataframe['discourse_id'].isin(id_list)]
    dataframe.to_csv(name + '.csv', encoding='utf-8', index=False)


df = pd.read_csv('./feedback-prize-effectiveness/train.csv', encoding='utf-8')
prompts = pd.read_csv('./clusters_effectivness.csv', encoding='utf-8')
df['prompt'] = df['essay_id'].map(prompts.set_index('id')['cluster'])
submission = pd.read_csv('./feedback-prize-effectiveness/sample_submission.csv', encoding='utf-8')
target_list = submission.columns[1:].tolist()
for col in target_list:
    df[col] = np.where(df['discourse_effectiveness'] == col, 1, 0)

get_slice('train', df)
get_slice('validation', df)
get_slice('test', df)
