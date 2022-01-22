import math
import logging
import os.path
import time
import sys
import argparse
import torch
import numpy as np
import pickle
import tqdm
from pathlib import Path

from model.tgn import TGN
from evaluation.evaluation import eval_wsdm_edge_prediction, test_wsdm_edge_prediction
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor, NeighborFinder, get_adj_list, MetaPathFinder
from utils.data_processing import compute_time_statistics, get_wsdm_data
import pandas as pd

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('--dataset_name', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='B')
parser.add_argument('--dataset_path', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='')

parser.add_argument('--time_range', type=float, default=200,
                    help='The time range for multi-step prediction')

parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=5, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=20, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')

parser.add_argument('--train_node_embedding', action='store_true',
                    help='Whether to train node embedding for each node')
parser.add_argument('--dim_node_embedding', type=int, default=80,
                    help='The dimension of the node embedding if train node embedding')
parser.add_argument('--use_raw_node_features', action='store_true',
                    help='Whether to use raw node features as part of input features')
parser.add_argument('--minibatch_size', type=int, default=1000,
                    help='experience replay batch')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--time_dim', type=int, default=10, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1,
                    help='Every how many batches to backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
    "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
    "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
    "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
Path("./results/").mkdir(parents=True, exist_ok=True)

args.step_length = 1
# args.prefix = f"lr{args.lr}_sl{args.step_length}_minibatchsize_{args.minibatch_size}"
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.dataset_name}.pth'
get_checkpoint_path = lambda \
        epoch: f'./saved_checkpoints/{args.prefix}-{args.dataset_name}-{epoch}.pth'
TEST_RESULT_PATH = f'./results/{args.prefix}-dataset{args.dataset_name}-'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, (train_data, train_edge_features), \
(eval_data, eval_edge_features, eval_labels), (test_data, test_edge_features) \
    = get_wsdm_data(dataset_name=args.dataset_name, dataset_path=args.dataset_path)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Initialize training neighbor finder to retrieve temporal graph
train_adj_filepath = os.path.join(args.dataset_path, f"{args.dataset_name}_training_adj_list.pkl")
if os.path.isfile(train_adj_filepath):
    with open(train_adj_filepath, "rb") as f:
        train_adj_list, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps = pickle.load(f)
else:
    train_adj_list = get_adj_list(train_data, max_node_idx=node_features.shape[0])
    node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps = \
        get_neighbor(train_adj_list)
    with open(train_adj_filepath, "wb") as f:
        pickle.dump([train_adj_list, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps], f)

train_ngh_finder = NeighborFinder(train_adj_list,
                                  args.uniform,
                                  seed=None,
                                  node_to_neighbors=node_to_neighbors,
                                  node_to_edge_idxs=node_to_edge_idxs,
                                  node_to_edge_timestamps=node_to_edge_timestamps)

# Compute time statistics for training data
dataset_filepath = os.path.join(args.dataset_path, f"{args.dataset_name}_training_statistics.pkl")
if os.path.isfile(dataset_filepath):
    with open(dataset_filepath, "rb") as f:
        mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
            pickle.load(f)
else:
    mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
        compute_time_statistics(train_data.sources, train_data.destinations, train_data.timestamps)
    with open(dataset_filepath, "wb") as f:
        pickle.dump([mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst], f)

for i in range(args.n_runs):
    results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
    Path("results/").mkdir(parents=True, exist_ok=True)

    # Initialize Model
    tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
              edge_features=train_edge_features, device=device, n_layers=NUM_LAYER,
              n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
              message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
              n_time_embedding_features=TIME_DIM,
              train_node_embedding=args.train_node_embedding,
              dim_node_embedding=args.dim_node_embedding,
              use_raw_node_features=args.use_raw_node_features,
              memory_update_at_start=not args.memory_update_at_end,
              embedding_module_type=args.embedding_module,
              message_function=args.message_function,
              aggregator_type=args.aggregator,
              memory_updater_type=args.memory_updater,
              n_neighbors=NUM_NEIGHBORS,
              mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
              mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
              use_destination_embedding_in_message=args.use_destination_embedding_in_message,
              use_source_embedding_in_message=args.use_source_embedding_in_message,
              dyrep=args.dyrep)

    tgn = tgn.to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)

    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / BATCH_SIZE)

    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)

    val_aucs = []
    epoch_times = []
    total_epoch_times = []
    train_losses = []

    early_stopper = EarlyStopMonitor(max_round=args.patience)

    for epoch in range(NUM_EPOCH):
        start_epoch = time.time()

        # Train using only training graph
        tgn.set_neighbor_finder(train_ngh_finder)
        m_loss = []

        logger.info('start {} epoch'.format(epoch))
        pbar = tqdm.trange(0, num_batch, desc='Start training.')
        for batch_idx in pbar:
            loss = 0
            optimizer.zero_grad()

            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(num_instance, start_idx + BATCH_SIZE)
            sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                                train_data.destinations[start_idx:end_idx]
            edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
            timestamps_batch = train_data.timestamps[start_idx:end_idx]

            size = len(sources_batch)
            _, negatives_batch = train_rand_sampler.sample(size)

            tgn = tgn.train()

            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float32, device=device)
                neg_label = torch.zeros(size, dtype=torch.float32, device=device)
                labels = torch.cat([pos_label, neg_label], dim=0)

            inp_node_batch = np.concatenate(
                [sources_batch, destinations_batch, negatives_batch], axis=0)
            inp_timestamps_batch = np.concatenate(
                [timestamps_batch, timestamps_batch, timestamps_batch], axis=0)
            inp_edge_idxs_batch = np.concatenate(
                [edge_idxs_batch, edge_idxs_batch, edge_idxs_batch], axis=0)

            node_embeddings = tgn.compute_batch_temporal_embeddings(
                inp_node_batch, inp_timestamps_batch, inp_edge_idxs_batch, NUM_NEIGHBORS)

            source_node_embeddings = node_embeddings[:size]
            destination_node_embeddings = node_embeddings[size: 2*size]
            negative_node_embeddings = node_embeddings[2*size: 3*size]
            node_pair_source_embeddings = torch.cat([
                source_node_embeddings, source_node_embeddings], dim=0)
            node_pair_destination_embeddings = torch.cat(
                [destination_node_embeddings, negative_node_embeddings], dim=0)

            scores1 = tgn.affinity_score(node_pair_source_embeddings, node_pair_destination_embeddings).sigmoid()
            scores2 = tgn.affinity_score(node_pair_destination_embeddings, node_pair_source_embeddings).sigmoid()
            scores = torch.maximum(scores1, scores2)

            logits_loss = criterion(scores.squeeze(), labels) * 2  # to make the loss same to the original

            loss += logits_loss

            pbar.set_description(f"logits loss is {logits_loss.item():.5f}")

            loss.backward()
            optimizer.step()
            m_loss.append(loss.item())


        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)

        ### Validation
        val_ap, val_auc = eval_wsdm_edge_prediction(model=tgn,
                                                    data=eval_data,
                                                    eval_labels=eval_labels,
                                                    n_neighbors=NUM_NEIGHBORS)

        val_aucs.append(val_auc)
        train_losses.append(np.mean(m_loss))

        # Save temporary results to disk
        pickle.dump({
            "val_aucs": val_aucs,
            "train_losses": train_losses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        total_epoch_time = time.time() - start_epoch
        total_epoch_times.append(total_epoch_time)

        logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
        logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
        logger.info('val auc: {}, val ap: {}'.format(val_auc, val_ap))

        predictions = test_wsdm_edge_prediction(model=tgn,
                                                data=test_data,
                                                n_neighbors=NUM_NEIGHBORS)

        pd.DataFrame(predictions).to_csv(TEST_RESULT_PATH + f"auc_{val_aucs[-1]}--epoch_{epoch}.csv", header=False, index=False)