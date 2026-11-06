import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import NeighborSampler
from models.modules import TimeEncoder

class GCN(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        support = torch.matmul(input_features, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class GIKT_Encoder(nn.Module):
    def __init__(self, num_q, num_skill, embed_dim, hidden_dim, graph_type='association', device='cpu'):
        super(GIKT_Encoder, self).__init__()
        self.device = device
        self.graph_type = graph_type
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.q_embed = nn.Embedding(num_q + 1, embed_dim, padding_idx=0)
        self.c_embed = nn.Embedding(num_skill + 1, embed_dim, padding_idx=0)
        self.r_embed = nn.Embedding(2, embed_dim)

        self.interaction_embed = nn.Linear(embed_dim * 2, hidden_dim)

        self.gcn = GCN(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def build_local_adj(self, c_seq_detached, graph_type='association'):
        B, S = c_seq_detached.shape
        adj = torch.zeros(B, S, S, device=self.device)

        if graph_type == 'association':
            c_seq_expanded = c_seq_detached.unsqueeze(2)
            c_seq_transposed = c_seq_detached.unsqueeze(1)
            adj = (c_seq_expanded == c_seq_transposed).float()

        elif graph_type == 'transition':
            for i in range(1, S):
                mask = (c_seq_detached[:, i] == c_seq_detached[:, i-1]) & (c_seq_detached[:, i] > 0)
                adj[torch.arange(B), i, i-1] = mask.float()
                adj[torch.arange(B), i-1, i] = mask.float()

        identity_mask = torch.eye(S, device=self.device).unsqueeze(0)
        adj = torch.clamp(adj + identity_mask, 0, 1)

        deg = torch.sum(adj, dim=-1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)

        return adj_norm

    def forward(self, q_seq, c_seq, r_seq):
        B, S = q_seq.shape

        q_emb_seq = self.q_embed(q_seq)
        c_emb_seq = self.c_embed(c_seq)
        r_emb_seq = self.r_embed(r_seq)

        x_interaction = torch.cat([c_emb_seq, r_emb_seq], dim=-1)
        x_features = torch.tanh(self.interaction_embed(x_interaction))

        adj = self.build_local_adj(c_seq.detach(), self.graph_type)

        h_seq = self.gcn(x_features, adj)
        h_seq = self.layer_norm(self.dropout(h_seq) + x_features)

        return h_seq, q_emb_seq


class GIKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16,
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 embed_dim=64,
                 hidden_dim=64,
                 graph_type='association'):

        super(GIKT, self).__init__()

        self.num_neighbors = num_neighbors
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.skill_map = self.node_raw_features[:, 0].long().to(device)

        self.dropout = dropout
        self.device = device

        self.model_encoder = GIKT_Encoder(
            num_q=self.num_nodes,
            num_skill=self.num_skills,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            graph_type=graph_type,
            device=device
        )
        self.model_encoder.to(device)


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def get_features(self, q_seq_np: np.ndarray, r_seq_np: np.ndarray):
        q_seq_np[q_seq_np < 0] = 0
        r_seq_np[r_seq_np < 0] = 0

        q_seq = torch.from_numpy(q_seq_np).long().to(self.device)
        r_seq = torch.from_numpy(r_seq_np).long().to(self.device)

        q_seq = torch.clamp(q_seq, 0, self.num_nodes - 1)
        r_seq = torch.clamp(r_seq, 0, 1)

        c_seq = self.skill_map[q_seq]

        return q_seq, c_seq, r_seq


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)

        q_seq_np = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)

        r_seq_np = np.concatenate(
            (src_neighbor_edge_ids, edge_ids[:, np.newaxis]), axis=1)

        q_seq, c_seq, r_seq = self.get_features(q_seq_np, r_seq_np)

        h_seq, q_emb_seq = self.model_encoder(q_seq, c_seq, r_seq)

        src_node_embeddings = h_seq[:, -1, :]
        dst_node_embeddings = q_emb_seq[:, -1, :]

        return src_node_embeddings, dst_node_embeddings