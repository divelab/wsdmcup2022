import os
import torch
import random
import numpy as np
import pandas as pd


class Data:
    def __init__(self, sources, destinations, timestamps, edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.edge_idxs = edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def relabel(self):
        pass


class EvalData:
    def __init__(self, sources, destinations, start_timestamps, end_timestamps, eval_edge_idxs, labels):
        self.sources = sources
        self.destinations = destinations
        self.start_timestamps = start_timestamps
        self.end_timestamps = end_timestamps
        self.eval_edge_idxs = eval_edge_idxs
        self.labels = labels
        self.n_interactions = len(sources)
        self.unique_nodes = set(sources) | set(destinations)
        self.n_unique_nodes = len(self.unique_nodes)

    def relabel(self):
        pass


# we keep the same node_idx, and add -1 node embeddings for the unnecessary nodes
def get_wsdm_data(dataset_path, dataset_name):
    if dataset_name == 'A':
        edge_csv = pd.read_csv(os.path.join(dataset_path, f"edges_train_{dataset_name}_reindex.csv"), header=None)
        train_timestamps = edge_csv.iloc[:, 3].to_numpy()
        record_time = np.unique(train_timestamps)

        START_TIME = record_time[0]
        RECORD_STEP = record_time[1] - record_time[0]
        train_timestamps = (train_timestamps - START_TIME) / RECORD_STEP
        # for A, from 0 to 22512
        edge_idxs = np.array(edge_csv.index)
        START_EDGE_IDX = edge_idxs[0]
        train_data = Data(edge_csv.iloc[:, 0].to_numpy(),
                          edge_csv.iloc[:, 1].to_numpy(),
                          train_timestamps,
                          edge_idxs - START_EDGE_IDX,
                          edge_csv.iloc[:, 2].to_numpy())

        # Assign Node feature in to graph
        # total_nodes = 69992
        total_nodes = 19442
        node_fea_csv = pd.read_csv(os.path.join(dataset_path, "node_features_reindex.csv"), header=None)
        edge_feature_csv = pd.read_csv(os.path.join(dataset_path, "edge_type_features.csv"), header=None)

        ori_node_features = node_fea_csv.iloc[:, 1:].to_numpy()
        node_idx = node_fea_csv.iloc[:, 0].to_numpy()
        node_features = np.ones((total_nodes, ori_node_features.shape[-1])) * (-1)
        node_features[node_idx] = ori_node_features

        eval_csv = pd.read_csv(os.path.join(dataset_path, "input_A_initial_reindex.csv"), header=None)
        start_eval_timestamps = (eval_csv.iloc[:, 3].to_numpy() - START_TIME) / RECORD_STEP
        end_eval_timestamps = (eval_csv.iloc[:, 4].to_numpy() - START_TIME) / RECORD_STEP
        eval_data = EvalData(eval_csv.iloc[:, 0].to_numpy(),
                             eval_csv.iloc[:, 1].to_numpy(),
                             start_eval_timestamps,
                             end_eval_timestamps,
                             np.array(eval_csv.index),
                             eval_csv.iloc[:, 2].to_numpy())

        # with open('destinations.csv', 'rb') as fd:
        #     gzip_fd = gzip.GzipFile(fileobj=fd)
        #     destinations = pd.read_csv(gzip_fd, header=None)
        #     destinations.to_csv('destinations.csv', header=False, index=False)

        test_csv = pd.read_csv(os.path.join(dataset_path, "input_A_reindex.csv"), header=None)
        start_test_timestamps = (test_csv.iloc[:, 3].to_numpy() - START_TIME) / RECORD_STEP
        end_test_timestamps = (test_csv.iloc[:, 4].to_numpy() - START_TIME) / RECORD_STEP
        test_data = EvalData(test_csv.iloc[:, 0].to_numpy(),
                             test_csv.iloc[:, 1].to_numpy(),
                             start_test_timestamps,
                             end_test_timestamps,
                             np.array(test_csv.index),
                             test_csv.iloc[:, 2].to_numpy())

        train_edge_features = edge_feature_csv.loc[:, 1:].loc[train_data.labels].to_numpy()
        eval_edge_features = edge_feature_csv.loc[:, 1:].loc[eval_data.labels].to_numpy()
        eval_labels = eval_csv.iloc[:, 5].to_numpy()
        test_edge_features = edge_feature_csv.loc[:, 1:].loc[test_data.labels].to_numpy()

    elif dataset_name == 'B':
        edge_csv = pd.read_csv(os.path.join(dataset_path, f"edges_train_{dataset_name}.csv"), header=None)
        train_timestamps = edge_csv.iloc[:, 3].to_numpy()
        record_time = np.unique(train_timestamps)

        total_nodes = max(edge_csv.iloc[:, 0].max(), edge_csv.iloc[:, 1].max()) + 1
        train_timestamps = edge_csv.iloc[:, 3].to_numpy()
        START_TIME = np.unique(train_timestamps).min()
        RECORD_STEP = 3600
        train_timestamps = (train_timestamps - START_TIME) / RECORD_STEP

        edge_idxs = np.array(edge_csv.index)
        train_data = Data(edge_csv.iloc[:, 0].to_numpy(),
                          edge_csv.iloc[:, 1].to_numpy(),
                          train_timestamps,
                          edge_idxs,
                          edge_csv.iloc[:, 2].to_numpy())
        node_features = np.arange(total_nodes).reshape(total_nodes, 1).astype(np.float32)

        eval_csv = pd.read_csv(os.path.join(dataset_path, "input_B_initial.csv"), header=None)
        start_eval_timestamps = (eval_csv.iloc[:, 3].to_numpy() - START_TIME) / RECORD_STEP
        end_eval_timestamps = (eval_csv.iloc[:, 4].to_numpy() - START_TIME) / RECORD_STEP
        eval_data = EvalData(eval_csv.iloc[:, 0].to_numpy(),
                             eval_csv.iloc[:, 1].to_numpy(),
                             start_eval_timestamps,
                             end_eval_timestamps,
                             np.array(eval_csv.index),
                             eval_csv.iloc[:, 2].to_numpy())
        eval_labels = eval_csv.iloc[:, 5]

        test_csv = pd.read_csv(os.path.join(dataset_path, 'input_B.csv'), header=None)
        start_test_timestamps = (test_csv.iloc[:, 3].to_numpy() - START_TIME) / RECORD_STEP
        end_test_timestamps = (test_csv.iloc[:, 4].to_numpy() - START_TIME) / RECORD_STEP
        test_data = EvalData(test_csv.iloc[:, 0].to_numpy(),
                             test_csv.iloc[:, 1].to_numpy(),
                             start_test_timestamps,
                             end_test_timestamps,
                             np.array(test_csv.index),
                             test_csv.iloc[:, 2].to_numpy())

        total_edge_type = np.unique(np.concatenate([train_data.labels, eval_data.labels, test_data.labels]))
        edge_features = np.eye(total_edge_type.max() + 1)
        train_edge_features = edge_features[train_data.labels]
        eval_edge_features = edge_features[eval_data.labels]
        test_edge_features = edge_features[test_data.labels]

    else:
        raise ValueError(f"dataset {dataset_name} is not implemented")

    return node_features, (train_data, train_edge_features), \
           (eval_data, eval_edge_features, eval_labels), \
           (test_data, test_edge_features)


def compute_time_statistics(sources, destinations, timestamps):
    START_TIME = timestamps.min()
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        # set the start time
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = START_TIME
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = START_TIME
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)
    # record the time shift mean and std for each nodes (as source or dst nodes)
    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)

    return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
