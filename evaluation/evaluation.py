import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_wsdm_edge_prediction(model, data, n_neighbors, eval_labels, batch_size=200):
    val_ap, val_auc = [], []
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            start_timestamps_batch = data.start_timestamps[s_idx:e_idx]
            end_timestamps_batch = data.end_timestamps[s_idx: e_idx]
            edge_idxs_batch = data.eval_edge_idxs[s_idx: e_idx]

            time_intepolates = [(end_timestamps_batch - start_timestamps_batch) * i / 14 + start_timestamps_batch for i
                                in range(15)]

            sources_node_embeddings = [model.compute_batch_temporal_embeddings(
                sources_batch, time, edge_idxs_batch, n_neighbors) for time in time_intepolates]
            destinations_node_embeddings = [model.compute_batch_temporal_embeddings(
                destinations_batch, time, edge_idxs_batch, n_neighbors) for time in time_intepolates]

            scores = [model.affinity_score(sources_node_embedding, destinations_node_embedding)
                      for sources_node_embedding, destinations_node_embedding in
                      zip(sources_node_embeddings, destinations_node_embeddings)]

            score = torch.stack(scores, dim=0)

            if score.shape[-1] > 1:
                pos_prob = score[:, :, 1]
            else:
                pos_prob = score

            pred_score = pos_prob.max(dim=0)[0].cpu().numpy()
            true_label = eval_labels[s_idx:e_idx]

            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_ap), np.mean(val_auc)


def test_wsdm_edge_prediction(model, data, n_neighbors, batch_size=200):
    pred_probs = []
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            start_timestamps_batch = data.start_timestamps[s_idx:e_idx]
            end_timestamps_batch = data.end_timestamps[s_idx: e_idx]
            edge_idxs_batch = data.eval_edge_idxs[s_idx: e_idx]

            time_intepolates = [(end_timestamps_batch - start_timestamps_batch) * i / 14 + start_timestamps_batch for i
                                in range(15)]

            pos_prob = [model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                         None, time,
                                                         edge_idxs_batch, n_neighbors)[0] for time in time_intepolates]

            pos_prob = torch.stack(pos_prob, dim=0).max(dim=0)[0]

            pred_score = pos_prob.cpu().numpy()
            pred_probs.append(pred_score)

    return np.concatenate(pred_probs, axis=0)
