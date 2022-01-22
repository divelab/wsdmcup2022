import pandas as pd

data_path = ''

node_feat_csv = pd.read_csv(data_path + '/node_features.csv', header=None,
                            names=['idx', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8'])
node_idx = node_feat_csv['idx'].values
node_map = {}
for i, idx in enumerate(node_idx):
    node_map[idx] = i
node_feat_csv['idx'] = node_feat_csv['idx'].map(node_map)

node_feat_csv.to_csv(data_path + '/node_features_reindex.csv', index=False, header=None)

edge_csv = pd.read_csv(data_path + '/edges_train_A.csv', header=None, names=['u', 'i', 'idx', 'ts'])
edge_csv['u'] = edge_csv['u'].map(node_map)
edge_csv['i'] = edge_csv['i'].map(node_map)
edge_csv.to_csv(data_path + '/edges_train_A_reindex.csv', index=False, header=None)

edge_csv = pd.read_csv(data_path + '/input_A_initial.csv', header=None,
                       names=['u', 'i', 'idx', 'ts1', 'ts2', 'label'])
edge_csv['u'] = edge_csv['u'].map(node_map)
edge_csv['i'] = edge_csv['i'].map(node_map)
edge_csv.to_csv(data_path + '/input_A_initial_reindex.csv', index=False, header=None)

edge_csv = pd.read_csv(data_path + '/input_A.csv', header=None, names=['u', 'i', 'idx', 'ts1', 'ts2'])
edge_csv['u'] = edge_csv['u'].map(node_map)
edge_csv['i'] = edge_csv['i'].map(node_map)
edge_csv.to_csv(data_path + '/input_A_reindex.csv', index=False, header=None)