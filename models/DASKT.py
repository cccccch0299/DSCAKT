import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import NeighborSampler

class DenseLocalGAT(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.2, concat=True):
        super(DenseLocalGAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        
        self.head_dim = out_channels // heads if concat else out_channels
        
        self.lin = nn.Linear(in_channels, self.head_dim * heads, bias=False)
        
        self.att_src = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, self.head_dim))
        
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, mask=None):
        B, S, _ = x.shape
        
        x_feat = self.lin(x).view(B, S, self.heads, self.head_dim)
        
        alpha_src = (x_feat * self.att_src).sum(dim=-1)
        alpha_dst = (x_feat * self.att_dst).sum(dim=-1)
        
        score_center = alpha_src + alpha_dst
        feat_center = x_feat
        
        alpha_src_pad = F.pad(alpha_src, (0, 0, 1, 0), "constant", 0)
        alpha_dst_pad = F.pad(alpha_dst, (0, 0, 1, 0), "constant", 0)
        x_feat_pad = F.pad(x_feat, (0, 0, 0, 0, 1, 0), "constant", 0)
        
        score_left = alpha_src + alpha_dst_pad[:, :-1, :]
        feat_left = x_feat_pad[:, :-1, :, :]
        
        alpha_dst_pad_r = F.pad(alpha_dst, (0, 0, 0, 1), "constant", 0)
        x_feat_pad_r = F.pad(x_feat, (0, 0, 0, 0, 0, 1), "constant", 0)
        
        score_right = alpha_src + alpha_dst_pad_r[:, 1:, :]
        feat_right = x_feat_pad_r[:, 1:, :, :]
        
        scores = torch.stack([score_left, score_center, score_right], dim=2)
        scores = self.leaky_relu(scores)
        
        attn = F.softmax(scores, dim=2) 
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        neighbor_feats = torch.stack([feat_left, feat_center, feat_right], dim=2)
        
        out = (attn.unsqueeze(-1) * neighbor_feats).sum(dim=2)
        
        if self.concat:
            out = out.view(B, S, self.heads * self.head_dim)
        else:
            out = out.mean(dim=2)
            
        return out

class DASKT_Model(nn.Module):
    def __init__(self, num_p, num_q, emb_size, dropout=0.2):
        super(DASKT_Model, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = emb_size
        
        self.lstm_layer = nn.LSTM(self.emb_size * 4, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.embedding_problem = nn.Embedding(num_p + 1, emb_size, padding_idx=0)
        self.embedding_question = nn.Embedding(num_q + 1, emb_size, padding_idx=0)
        self.answer_embedding = nn.Embedding(2, emb_size)
        
        self.affect_projector = nn.Linear(2, emb_size) 
        
        self.affconcat_layer = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.conv1 = DenseLocalGAT(self.hidden_size, 128, heads=8, dropout=0.2, concat=True)
        
        self.conv2 = DenseLocalGAT(128 * 8, self.hidden_size, heads=1, dropout=0.2, concat=False)

    def forward(self, p, q, r, t):
        p_emb = self.embedding_problem(p)
        q_emb = self.embedding_question(q)
        r_emb = self.answer_embedding(r)
        
        t_gap = torch.zeros_like(t, dtype=torch.float)
        t_gap[:, 1:] = (t[:, 1:] - t[:, :-1]).float()
        t_gap = torch.log(t_gap + 1.0)
        
        aff_raw = torch.stack([r.float(), t_gap], dim=-1)
        aff_emb = torch.tanh(self.affect_projector(aff_raw))

        aff_con_data = torch.cat((p_emb, aff_emb), dim=-1)
        
        aff_con_data_fc = self.affconcat_layer(aff_con_data)
        
        x = F.dropout(aff_con_data_fc, p=0.6, training=self.training)
        
        x = F.elu(self.conv1(x))
        
        x = F.dropout(x, p=0.6, training=self.training)
        
        aff_final = self.conv2(x)

        concatenated_qr = torch.cat((p_emb, q_emb, r_emb, aff_final), dim=-1)

        h, _ = self.lstm_layer(concatenated_qr)

        h = self.dropout_layer(h)

        return h, p_emb, q_emb

class DASKT(nn.Module):
    def __init__(self, node_raw_features: np.ndarray,
                 edge_raw_features: np.ndarray,
                 time_dim=16, 
                 num_neighbors: int = 20,
                 dropout: float = 0.2,
                 device: str = 'cuda:0',
                 emb_size=64):
        
        super(DASKT, self).__init__()
        
        self.num_neighbors = num_neighbors
        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]
        
        self.skill_map = self.node_raw_features[:, 0].long().to(device)
        
        self.device = device
        
        self.daskt_model = DASKT_Model(
            num_p=self.num_nodes,
            num_q=self.num_skills,
            emb_size=emb_size,
            dropout=dropout
        )
        self.daskt_model.to(device)
        self.neighbor_sampler = None

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_embeddings(self, src_node_ids: np.ndarray, edge_ids: np.ndarray,
                                                 node_interact_times: np.ndarray, dst_node_ids: np.ndarray):
        
        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = self.neighbor_sampler.get_historical_neighbors(
            src_node_ids, node_interact_times, self.num_neighbors)
        
        p_seq_np = np.concatenate((src_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1)
        r_seq_np = np.concatenate((src_neighbor_edge_ids, np.zeros((len(src_node_ids), 1)).astype(np.longlong)), axis=1)
        t_seq_np = np.concatenate((src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1)

        p_seq = torch.from_numpy(p_seq_np).long().to(self.device)
        r_seq = torch.from_numpy(r_seq_np).long().to(self.device)
        t_seq = torch.from_numpy(t_seq_np).long().to(self.device)
        
        q_seq = self.skill_map[p_seq]
        
        h, p_emb_seq, q_emb_seq = self.daskt_model(p_seq, q_seq, r_seq, t_seq)
        
        src_node_embeddings = h[:, -1, :]
        
        p_next = p_emb_seq[:, -1, :]
        q_next = q_emb_seq[:, -1, :]
        dst_node_embeddings = p_next + q_next

        return src_node_embeddings, dst_node_embeddings