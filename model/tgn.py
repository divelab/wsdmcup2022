import logging
import os.path

import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modelmodules.memory import Memory
from modelmodules.message_aggregator import get_message_aggregator
from modelmodules.message_function import get_message_function
from modelmodules.memory_updater import get_memory_updater
from modelmodules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, output_channel=1,
                 n_layers=2, n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True, message_dimension=100,
                 n_time_embedding_features=10,
                 memory_dimension=500, use_raw_node_features=False,
                 train_node_embedding=False, dim_node_embedding=100,
                 embedding_module_type="graph_attention", message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1, mean_time_shift_dst=0,
                 std_time_shift_dst=1, n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 dyrep=False):
        super(TGN, self).__init__()

        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.output_channel = output_channel

        self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

        self.n_time_embedding_features = n_time_embedding_features
        self.n_node_features = self.node_raw_features.shape[1]
        self.n_nodes = self.node_raw_features.shape[0]
        self.n_edge_features = self.edge_raw_features.shape[1]
        self.embedding_dimension = self.n_node_features
        self.n_neighbors = n_neighbors
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep

        self.use_memory = use_memory
        self.time_encoder = TimeEncode(dimension=self.n_time_embedding_features)
        self.memory = None

        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst

        if self.use_memory:
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                                    self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 input_dimension=message_dimension,
                                 message_dimension=message_dimension,
                                 device=device)
            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)
            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module_type = embedding_module_type
        # self.pre_lin = torch.nn.Linear(self.n_node_features, self.n_node_pre_lin_features)

        self.train_node_embedding = train_node_embedding  # pass it to embedding module
        self.use_raw_node_features = use_raw_node_features
        self.dim_node_embedding = dim_node_embedding
        if self.train_node_embedding:
            self.n_node_features = dim_node_embedding
            if self.use_raw_node_features:
                self.n_node_features += self.node_raw_features.shape[1]

        emb_in_node_features = self.n_node_features + self.memory_dimension if self.use_memory else self.n_node_features
        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     edge_features=self.edge_raw_features,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     use_raw_node_features=self.use_raw_node_features,
                                                     train_node_embedding=self.train_node_embedding,
                                                     dim_node_embedding=self.dim_node_embedding,
                                                     n_layers=self.n_layers,
                                                     n_node_features=emb_in_node_features,
                                                     n_edge_features=self.n_edge_features,
                                                     n_time_features=self.n_time_embedding_features,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors)

        # MLP to compute probability on an edge given two node embeddings
        # here we concatenate the time embedding with the node embedding instead of adding
        self.affinity_score = MergeLayer(emb_in_node_features, emb_in_node_features,
                                         emb_in_node_features,
                                         self.output_channel)

    def compute_temporal_embeddings(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                    edge_idxs, n_neighbors=20):
        """
        Compute temporal embeddings for sources, destinations, and negatively sampled destinations.
        source_nodes [batch_size]: source ids.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Temporal embeddings for sources, destinations and negatives
        """

        n_samples = len(source_nodes)
        if negative_nodes is None:
            nodes = np.concatenate([source_nodes, destination_nodes])
            positives = np.concatenate([source_nodes, destination_nodes])
            timestamps = np.concatenate([edge_times, edge_times])
        else:
            nodes = np.concatenate([source_nodes, destination_nodes, negative_nodes])
            positives = np.concatenate([source_nodes, destination_nodes])
            timestamps = np.concatenate([edge_times, edge_times, edge_times])

        memory = None
        time_diffs = None
        if self.use_memory:
            if self.memory_update_at_start and self.training:
                # Update memory for all nodes with messages stored in previous batches
                memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                              self.memory.messages)
            else:
                memory = self.memory.get_memory(list(range(self.n_nodes)))
                last_update = self.memory.last_update

            ### Compute differences between the time the memory of a node was last updated,
            ### and the time for which we want to compute the embedding of a node
            source_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[
                source_nodes]
            source_time_diffs = (source_time_diffs - self.mean_time_shift_src) / self.std_time_shift_src
            destination_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[
                destination_nodes]
            destination_time_diffs = (destination_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
            if negative_nodes is None:
                time_diffs = torch.cat([source_time_diffs, destination_time_diffs], dim=0)
            else:
                negative_time_diffs = torch.FloatTensor(edge_times).to(self.device) - last_update[
                    negative_nodes]
                negative_time_diffs = (negative_time_diffs - self.mean_time_shift_dst) / self.std_time_shift_dst
                time_diffs = torch.cat([source_time_diffs, destination_time_diffs, negative_time_diffs], dim=0)

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        if negative_nodes is None:
            negative_node_embedding = None
        else:
            negative_node_embedding = node_embedding[2 * n_samples:]

        if self.use_memory and self.training:
            if self.memory_update_at_start:
                # Persist the updates to the memory only for sources and destinations (since now we have
                # new messages for them)
                self.update_memory(positives, self.memory.messages)

                assert torch.allclose(memory[positives], self.memory.get_memory(positives), atol=1e-5), \
                    "Something wrong in how the memory was updated"

                # Remove messages for the positives since we have already updated the memory using them
                self.memory.clear_messages(positives)

            unique_sources, source_id_to_messages = self.get_raw_messages(source_nodes,
                                                                          source_node_embedding,
                                                                          destination_nodes,
                                                                          destination_node_embedding,
                                                                          edge_times, edge_idxs)
            unique_destinations, destination_id_to_messages = self.get_raw_messages(destination_nodes,
                                                                                    destination_node_embedding,
                                                                                    source_nodes,
                                                                                    source_node_embedding,
                                                                                    edge_times, edge_idxs)
            if self.memory_update_at_start:
                self.memory.store_raw_messages(unique_sources, source_id_to_messages)
                self.memory.store_raw_messages(unique_destinations, destination_id_to_messages)
            else:
                self.update_memory(unique_sources, source_id_to_messages)
                self.update_memory(unique_destinations, destination_id_to_messages)

        if self.use_memory and self.dyrep:
            source_node_embedding = memory[source_nodes]
            destination_node_embedding = memory[destination_nodes]
            if negative_nodes is None:
                negative_node_embedding = None
            else:
                negative_node_embedding = memory[negative_nodes]

        return source_node_embedding, destination_node_embedding, negative_node_embedding

    def compute_batch_temporal_embeddings(self, nodes, timestamps, edge_idxs, n_neighbors=20):
        memory = None
        time_diffs = None

        # Compute the embeddings using the embedding module
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=n_neighbors,
                                                                 time_diffs=time_diffs)

        return node_embedding

    def compute_edge_probabilities(self, source_nodes, destination_nodes, negative_nodes, edge_times,
                                   edge_idxs, n_neighbors=20):
        """
        Compute probabilities for edges between sources and destination and between sources and
        negatives by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        :param destination_nodes [batch_size]: destination ids
        :param negative_nodes [batch_size]: ids of negative sampled destination
        :param edge_times [batch_size]: timestamp of interaction
        :param edge_idxs [batch_size]: index of interaction
        :param n_neighbors [scalar]: number of temporal neighbor to consider in each convolutional
        layer
        :return: Probabilities for both the positive and negative edges
        """
        # use this when the output channel equal to 1
        n_samples = len(source_nodes)
        # change compute_temporal_embeddings into a multi-pairs way.
        source_node_embedding, destination_node_embedding, negative_node_embedding = self.compute_temporal_embeddings(
            source_nodes, destination_nodes, negative_nodes, edge_times, edge_idxs, n_neighbors)

        if negative_nodes is not None:
            source_embedding = torch.cat([source_node_embedding, source_node_embedding], dim=0),
            destination_embedding = torch.cat([destination_node_embedding, negative_node_embedding])
            score = torch.maximum(self.affinity_score(source_embedding, destination_embedding),
                                  self.affinity_score(destination_embedding, source_embedding)).squeeze(dim=0)
            pos_score = score[:n_samples]
            neg_score = score[n_samples:]
        else:
            score = torch.maximum(self.affinity_score(source_node_embedding, destination_node_embedding),
                                  self.affinity_score(destination_node_embedding, source_node_embedding))

            pos_score = score[:n_samples]
            neg_score = pos_score
        return pos_score.sigmoid(), neg_score.sigmoid(), (source_node_embedding, destination_node_embedding, negative_node_embedding)

    def update_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        # Update the memory with the aggregated messages
        self.memory_updater.update_memory(unique_nodes, unique_messages,
                                          timestamps=unique_timestamps)

    def get_updated_memory(self, nodes, messages):
        # Aggregate messages for the same nodes
        unique_nodes, unique_messages, unique_timestamps = \
            self.message_aggregator.aggregate(
                nodes,
                messages)

        if len(unique_nodes) > 0:
            unique_messages = self.message_function.compute_message(unique_messages)

        updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                     unique_messages,
                                                                                     timestamps=unique_timestamps)

        return updated_memory, updated_last_update

    def get_raw_messages(self, source_nodes, source_node_embedding, destination_nodes,
                         destination_node_embedding, edge_times, edge_idxs):
        edge_times = torch.from_numpy(edge_times).float().to(self.device)
        edge_features = self.edge_raw_features[edge_idxs]

        source_memory = self.memory.get_memory(source_nodes) if not \
            self.use_source_embedding_in_message else source_node_embedding
        destination_memory = self.memory.get_memory(destination_nodes) if \
            not self.use_destination_embedding_in_message else destination_node_embedding

        source_time_delta = edge_times - self.memory.last_update[source_nodes]
        source_time_delta_encoding = self.time_encoder(source_time_delta.unsqueeze(dim=1)).view(len(
            source_nodes), -1)

        source_message = torch.cat([source_memory, destination_memory, edge_features,
                                    source_time_delta_encoding],
                                   dim=1)
        messages = defaultdict(list)
        unique_sources = np.unique(source_nodes)

        for i in range(len(source_nodes)):
            messages[source_nodes[i]].append((source_message[i], edge_times[i]))

        return unique_sources, messages

    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder