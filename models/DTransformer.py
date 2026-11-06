import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.utils import NeighborSampler

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DTransformer_Core(nn.Module):
    def __init__(self, num_q, num_skill, d_model, n_heads, n_layers, dropout, device='cpu'):
        super(DTransformer_Core, self).__init__()
        self.device = device
        self.d_model = d_model
        
        self.q_embed = nn.Embedding(num_q + 1, d_model, padding_idx=0)
        self.c_embed = nn.Embedding(num_skill + 1, d_model, padding_idx=0)
        self.r_embed = nn.Embedding(2, d_model)
        
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=200)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=n_layers)

        self.theta_ability = nn.Embedding(num_skill + 1, d_model, padding_idx=0)
        self.d_difficulty = nn.Embedding(num_q + 1, d_model, padding_idx=0)
        
        self.layer_norm = nn.LayerNorm(d_model)

    def _generate_causal_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def forward(self, q_seq, c_seq, r_seq, mask_seq):
        B, S_max = q_seq.shape

        q_emb_seq = self.q_embed(q_seq)
        c_emb_seq = self.c_embed(c_seq)
        r_emb_seq = self.r_embed(r_seq)

        theta = self.theta_ability(c_seq)
        d = self.d_difficulty(q_seq)
        
        x_in = r_emb_seq + d + theta
        x_in = self.pos_encoder(x_in)
        x_in = self.layer_norm(x_in)

        attn_mask = self._generate_causal_mask(S_max)
        padding_mask = (mask_seq == 0)

        h_seq = self.transformer_encoder(
            x_in, 
            mask=attn_mask, 
            src_key_padding_mask=padding_mask,
            is_causal=False
        ) 

        return h_seq, q_emb_seq


class DTransformer(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16, 
                 num_neighbors: int = 20,
                 dropout: float = 0.5,
                 device: str = 'cuda:0',
                 dataset_name: str = None,
                 d_model=64,
                 n_heads=8,
                 n_layers=2):
        
        super(DTransformer, self).__init__()
        
            
        self.num_neighbors = num_neighbors
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.skill_map = self.node_raw_features[:, 0].long().to(device)

        self.dropout = dropout
        self.device = device
        
        self.dataset_path = f'./processed_data/{dataset_name}/'
        
        try:
            u_f = np.load(self.dataset_path + 'u_seq.npy', allow_pickle=True)
            q_f = np.load(self.dataset_path + 'q_seq.npy', allow_pickle=True)
            c_f = np.load(self.dataset_path + 'c_seq.npy', allow_pickle=True)
            r_f = np.load(self.dataset_path + 'r_seq.npy', allow_pickle=True)
            t_f = np.load(self.dataset_path + 't_seq.npy', allow_pickle=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"error {self.dataset_path} not found (*_seq.npy)。")

        n_users = len(u_f) 
        if n_users == self.num_nodes:
             n_users = self.num_nodes - 1
        
        self.u_seq = [torch.from_numpy(u_f[i]).long().to(device) for i in range(n_users + 1)]
        self.q_seq = [torch.from_numpy(q_f[i]).long().to(device) for i in range(n_users + 1)]
        self.c_seq = [torch.from_numpy(c_f[i]).long().to(device) for i in range(n_users + 1)]
        self.r_seq = [torch.from_numpy(r_f[i]).long().to(device) for i in range(n_users + 1)]
        self.t_seq = [torch.from_numpy(t_f[i]).long().to(device) for i in range(n_users + 1)]

        self.tgt_node_features = self.node_raw_features.long().to(device)

        self.model_encoder = DTransformer_Core(
            num_q=self.num_nodes,
            num_skill=self.num_skills,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device
        )
        self.model_encoder.to(device)


    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        pass

    def get_features(self, u, t, q, edge_ids):
        seq_len = torch.tensor([len(self.u_seq[x]) for x in u], device=self.device)
        real_max_len = torch.max(seq_len) if seq_len.numel() > 0 else 0
        S_max = real_max_len + 1 
        
        q_seq = torch.zeros(len(u), S_max, device=self.device).long()
        c_seq = torch.zeros(len(u), S_max, device=self.device).long()
        r_seq = torch.zeros(len(u), S_max, device=self.device).long()
        mask_seq = torch.zeros(len(u), S_max, device=self.device).long()
        
        final_seq_lens = torch.zeros(len(u), device=self.device).long()

        for i, (u_i, t_i, q_i, r_i) in enumerate(zip(u, t, q, edge_ids)):
            idx = (self.t_seq[u_i] < t_i).nonzero(as_tuple=True)[0]
            
            hist_q = self.q_seq[u_i][idx]
            hist_c = self.c_seq[u_i][idx]
            hist_r = self.r_seq[u_i][idx]
            
            L = len(hist_q) + 1
            
            if L > S_max:
                hist_q = hist_q[L - S_max:]
                hist_c = hist_c[L - S_max:]
                hist_r = hist_r[L - S_max:]
                L = S_max
            
            if L > 1:
                q_seq[i, 0:L-1] = hist_q
                c_seq[i, 0:L-1] = hist_c
                r_seq[i, 0:L-1] = hist_r
            
            q_seq[i, L-1] = q_i
            c_seq[i, L-1] = self.tgt_node_features[q_i, 0].long() 
            r_seq[i, L-1] = r_i
            
            mask_seq[i, 0:L] = 1
            final_seq_lens[i] = L 

        return q_seq, c_seq, r_seq, mask_seq, final_seq_lens


    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        q_seq, c_seq, r_seq, mask, sl = self.get_features(src_node_ids, node_interact_times, dst_node_ids, edge_ids)

        h_seq, q_emb_seq = self.model_encoder(q_seq, c_seq, r_seq, mask)
        
        B = src_node_ids.shape[0]
        
        src_node_embeddings = h_seq[torch.arange(B), sl-1, :]
        dst_node_embeddings = q_emb_seq[torch.arange(B), sl-1, :]

        return src_node_embeddings, dst_node_embeddings