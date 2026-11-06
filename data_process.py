import pandas as pd
import numpy as np
import json

df = pd.read_csv('anonymized_full_release_competition_dataset.csv', low_memory=False)

student_id_counts = df['studentId'].value_counts()
students_to_keep = student_id_counts[student_id_counts > 5].index
problem_id_counts = df['problemId'].value_counts()
problems_to_keep = problem_id_counts[problem_id_counts > 5].index

df_filtered = df[
    df['studentId'].isin(students_to_keep) &
    df['problemId'].isin(problems_to_keep)
].copy()
df_filtered.sort_values(by='endTime', ascending=True, inplace=True)

df_filtered['skill_encoded'] = df_filtered['skill'].astype('category').cat.codes

max_u1 = df_filtered['studentId'].max()
max_skill = df_filtered['skill_encoded'].max()
max_i1 = df_filtered['problemId'].max()

df_filtered['problemId'] = df_filtered['problemId'].astype('int64')
df_filtered['skill_encoded'] = df_filtered['skill_encoded'].astype('int64')

df_filtered['studentId'] += max_skill + 1
df_filtered['problemId'] += max_u1 + max_skill + 2


skill_map = df_filtered.drop_duplicates(subset=['skill_encoded']).set_index('skill_encoded')['skill']
skill_nodes = pd.DataFrame({
    'node_id': skill_map.index,
    'node_type': 'skill',
    'original_name': skill_map.values
})
question_nodes = pd.DataFrame({
    'node_id': df_filtered['problemId'].unique(),
    'node_type': 'question',
    'original_name': df_filtered['problemId'].unique()
})
node_df = pd.concat([skill_nodes, question_nodes], ignore_index=True)
node_df.to_csv('nodes.csv', index=False)

unique_pairs = df_filtered[['problemId', 'skill_encoded']].drop_duplicates()
edges = [(pid, sid, 0) for pid, sid in unique_pairs.itertuples(index=False)]
with open('concept.json') as f:
    prerequisite_pairs = json.load(f)

for a, b in prerequisite_pairs:
    edges.append((a, b, 1))

edge_df = pd.DataFrame(edges, columns=['source_id', 'target_id', 'edge_type'])
edge_df.to_csv('edge.csv', index=False)

final_columns = ['studentId', 'problemId', 'endTime', 'correct', 'skill_encoded']
df_final = df_filtered[final_columns].copy()
df_final.rename(columns={
    'studentId': 'u',
    'problemId': 'i',
    'endTime': 'ts',
    'correct': 'label'
}, inplace=True)
df_final['idx'] = range(1, len(df_final) + 1)

max_i_np = df_final['i'].max()
n = int(max_i_np)
skill_array = np.zeros(n + 1, dtype=int)
skill_map = df_final.drop_duplicates(subset=['i']).set_index('i')['skill_encoded']
skill_array[skill_map.index] = skill_map.values
final_skill_array = skill_array.reshape(-1, 1)

label_numpy_array = df_final['label'].to_numpy().reshape(-1, 1)
ml_assist = np.vstack([[[0]], label_numpy_array])

np.save('ml_assist18_node.npy', final_skill_array)
np.save('ml_assist18.npy', ml_assist)

df_final.drop('skill_encoded', axis=1, inplace=True)
df_final.to_csv('ml_assist18.csv', index=True)