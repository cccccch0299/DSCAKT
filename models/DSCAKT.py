import datetime
import numpy as np
import torch
import torch.nn as nn
from utils.utils import NeighborSampler
from models.modules import TimeEncoder, TimeDualDecayEncoder, RelationalKGEncoder
import torch.fft as fft
from pathlib import Path

SAVE_MID_FEATURE = False


class KGColdStartEmbed(nn.Module):
    def __init__(self, num_kg_nodes, embed_dim, kg_id_map, max_use=0):
        super().__init__()
        self.kg_id_map = kg_id_map
        self.max_use = max_use
        self.embed = nn.Embedding(num_kg_nodes, embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())

        self.register_buffer(
            "usage_counter", torch.zeros(num_kg_nodes, dtype=torch.long)
        )

    def forward(self, q_node_ids, zero_feat):
        if self.max_use <= 0:
            return zero_feat

        device = zero_feat.device
        B, L, D = zero_feat.shape

        kg_ids = torch.tensor(
            [
                [self.kg_id_map.get(int(nid), 0) for nid in neighbors]
                for neighbors in q_node_ids
            ],
            dtype=torch.long,
            device=device,
        )

        life_mask = self.usage_counter[kg_ids] < self.max_use  # [B, L]
        self.usage_counter[kg_ids] += life_mask

        kg_embed = self.embed(kg_ids)  # [B, L, D]
        gate_in = torch.cat([zero_feat, kg_embed], dim=-1)
        gate_weight = self.gate(gate_in) * life_mask.unsqueeze(-1)  # [B, L, D]

        return zero_feat + gate_weight * kg_embed


class DSCAKT(nn.Module):
    def __init__(
        self,
        node_raw_features: np.ndarray,
        edge_raw_features: np.ndarray,
        knowledge_graph_edge_index: torch.Tensor,
        knowledge_graph_edge_type: torch.Tensor,
        num_kg_nodes: int,
        num_kg_relations: int,
        kg_id_map: dict,
        time_dim=16,
        num_neighbors: int = 50,
        ablation="-1",
        dropout: float = 0.5,
        device: str = "cuda:0",
        freq_gate_dim_ratio=0.25,
        freq_gate_activation="relu",
        norm_momentum=0.9,
        norm_epsilon=1e-5,
    ):
        super(DSCAKT, self).__init__()
        self.num_neighbors = num_neighbors
        self.ablation = ablation

        self.node_raw_features = torch.from_numpy(node_raw_features.astype(np.float32))
        self.edge_raw_features = torch.from_numpy(edge_raw_features.astype(np.float32))

        self.num_skills = int(np.unique(self.node_raw_features[:, 0]).max()) + 1
        self.num_nodes = self.node_raw_features.shape[0]

        self.edge_dim = 64
        self.node_dim = 64
        self.time_dim = time_dim
        self.register_buffer("knowledge_graph_edge_index", knowledge_graph_edge_index)
        self.register_buffer("knowledge_graph_edge_type", knowledge_graph_edge_type)
        self.kg_id_map = kg_id_map

        self.kg_cold_start = KGColdStartEmbed(
            num_kg_nodes=num_kg_nodes,
            embed_dim=self.node_dim,
            kg_id_map=kg_id_map,
            max_use=2,
        )

        self.projection_layer = nn.ModuleDict(
            {
                "feature_Linear": nn.Linear(
                    self.node_raw_features.shape[-1], self.node_dim, bias=True
                ),
                "edge": nn.Linear(1, self.node_dim, bias=True),
                "time": nn.Linear(self.time_dim, self.node_dim, bias=True),
                "struct": nn.Linear(1, self.node_dim, bias=True),
            }
        )

        self.output_layer = nn.Linear(self.node_dim, self.node_dim, bias=True)
        self.dropout = dropout
        self.device = device

        self.src_node_updater = DSCAKT_Seq(
            edge_dim=self.edge_dim,
            node_dim=self.node_dim,
            ablation=self.ablation,
            freq_gate_dim_ratio=0.25,
            freq_gate_activation="relu",
            norm_momentum=0.9,
            norm_epsilon=1e-5,
        )
        self.dst_node_updater = DSCAKT_Seq(
            edge_dim=self.edge_dim,
            node_dim=self.node_dim,
            ablation=self.ablation,
            freq_gate_dim_ratio=0.25,
            freq_gate_activation="relu",
            norm_momentum=0.9,
            norm_epsilon=1e-5,
        )

        if self.ablation == "dual":
            self.time_encoder = TimeEncoder(time_dim=self.time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim=self.time_dim)

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        self.neighbor_sampler = neighbor_sampler

    def compute_embeddings(
        self,
        src_node_ids: np.ndarray,
        edge_ids: np.ndarray,
        node_interact_times: np.ndarray,
        dst_node_ids: np.ndarray,
    ):

        src_neighbor_node_ids, src_neighbor_edge_ids, src_neighbor_times = (
            self.neighbor_sampler.get_historical_neighbors(
                src_node_ids, node_interact_times, self.num_neighbors
            )
        )
        dst_neighbor_node_ids, dst_neighbor_edge_ids, dst_neighbor_times = (
            self.neighbor_sampler.get_historical_neighbors(
                dst_node_ids, node_interact_times, self.num_neighbors
            )
        )

        src_neighbor_node_ids = np.concatenate(
            (src_neighbor_node_ids, src_node_ids[:, np.newaxis]), axis=1
        )
        src_neighbor_edge_ids = np.concatenate(
            (
                src_neighbor_edge_ids,
                np.zeros((len(src_node_ids), 1)).astype(np.longlong),
            ),
            axis=1,
        )
        src_neighbor_times = np.concatenate(
            (src_neighbor_times, node_interact_times[:, np.newaxis]), axis=1
        )

        dst_neighbor_node_ids = np.concatenate(
            (dst_neighbor_node_ids, dst_node_ids[:, np.newaxis]), axis=1
        )
        dst_neighbor_edge_ids = np.concatenate(
            (
                dst_neighbor_edge_ids,
                np.zeros((len(dst_node_ids), 1)).astype(np.longlong),
            ),
            axis=1,
        )
        dst_neighbor_times = np.concatenate(
            (dst_neighbor_times, node_interact_times[:, np.newaxis]), axis=1
        )

        src_nodes_neighbor_co_occurrence_features = (
            (
                torch.from_numpy(src_neighbor_node_ids[:, :-1])
                == torch.from_numpy(dst_node_ids)
                .unsqueeze(1)
                .repeat(1, self.num_neighbors)
            )
            .unsqueeze(-1)
            .float()
            .to(self.device)
        )
        dst_nodes_neighbor_co_occurrence_features = (
            (
                torch.from_numpy(dst_neighbor_node_ids[:, :-1])
                == torch.from_numpy(src_node_ids)
                .unsqueeze(1)
                .repeat(1, self.num_neighbors)
            )
            .unsqueeze(-1)
            .float()
            .to(self.device)
        )

        src_node_skill = (
            self.node_raw_features[torch.from_numpy(src_neighbor_node_ids)][:, :-1, 0]
            .long()
            .to(self.device)
        )
        dst_node_skill = (
            self.node_raw_features[torch.from_numpy(dst_neighbor_node_ids)][:, -1, 0]
            .long()
            .to(self.device)
            .unsqueeze(1)
            .repeat(1, self.num_neighbors)
        )

        src_nodes_neighbor_skill_features = (
            (src_node_skill == dst_node_skill).unsqueeze(-1).float()
        )
        a = 1
        if self.ablation == "counter":
            a = 0

        src_nodes_neighbor_struct_features = self.projection_layer["struct"](
            a * src_nodes_neighbor_co_occurrence_features
        )
        dst_nodes_neighbor_struct_features = self.projection_layer["struct"](
            a * dst_nodes_neighbor_co_occurrence_features
        )
        src_nodes_neighbor_skill_struct_features = self.projection_layer["struct"](
            a * src_nodes_neighbor_skill_features
        )

        (
            src_nodes_neighbor_node_raw_features,
            src_nodes_edge_raw_features,
            src_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            nodes_edge_ids=src_neighbor_edge_ids,
            nodes_neighbor_ids=src_neighbor_node_ids,
            nodes_neighbor_times=src_neighbor_times,
        )
        (
            dst_nodes_neighbor_node_raw_features,
            dst_nodes_edge_raw_features,
            dst_nodes_neighbor_time_features,
        ) = self.get_features(
            node_interact_times=node_interact_times,
            nodes_edge_ids=dst_neighbor_edge_ids,
            nodes_neighbor_ids=dst_neighbor_node_ids,
            nodes_neighbor_times=dst_neighbor_times,
        )

        src_nodes_features = (
            src_nodes_neighbor_node_raw_features
            + src_nodes_edge_raw_features
            + src_nodes_neighbor_time_features
        )
        dst_nodes_features = (
            dst_nodes_neighbor_node_raw_features
            + dst_nodes_edge_raw_features
            + dst_nodes_neighbor_time_features
        )

        dst_nodes_features = self.kg_cold_start(
            dst_neighbor_node_ids, dst_nodes_features
        )

        src_node_embeddings = (
            self.src_node_updater.update(
                src_nodes_features[:, :-1, :]
                + src_nodes_neighbor_skill_struct_features
                + src_nodes_neighbor_struct_features
            )
            + (src_nodes_edge_raw_features + src_nodes_neighbor_time_features)[
                :, -1, :
            ]  # + src_kg_embeds[:, -1, :]
        )

        if self.ablation in ["q_qid", "q_kid"]:
            dst_node_embeddings = dst_nodes_neighbor_node_raw_features[:, -1]
        else:
            dst_node_embeddings = (
                self.dst_node_updater.update(
                    (dst_nodes_edge_raw_features + dst_nodes_neighbor_time_features)[
                        :, :-1, :
                    ]
                    + dst_nodes_neighbor_struct_features
                )
                + dst_nodes_features[:, -1, :]
            )

        src_node_embeddings = self.output_layer(src_node_embeddings)
        dst_node_embeddings = self.output_layer(dst_node_embeddings)

        return src_node_embeddings, dst_node_embeddings

    def get_features(
        self,
        nodes_neighbor_ids: np.ndarray,
        nodes_edge_ids: np.ndarray,
        nodes_neighbor_times: np.ndarray,
        node_interact_times: np.ndarray,
    ):
        if self.ablation in ["embed", "q_kid"]:
            nodes_neighbor_node_raw_features = self.projection_layer["feature_Embed"](
                self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)]
                .to(self.device)[:, :, 0]
                .long()
            )
        elif self.ablation == "q_qid":
            nodes_neighbor_node_raw_features = self.projection_layer["node"](
                torch.from_numpy(nodes_neighbor_ids).to(self.device)
            )
        else:
            nodes_neighbor_node_raw_features = self.projection_layer["feature_Linear"](
                self.node_raw_features[torch.from_numpy(nodes_neighbor_ids)].to(
                    self.device
                )
            )

        if self.ablation == "dual":
            nodes_neighbor_time_features = self.time_encoder(
                torch.from_numpy(
                    node_interact_times[:, np.newaxis] - nodes_neighbor_times
                )
                .float()
                .to(self.device)
            )
        else:
            nodes_neighbor_time_features = self.time_encoder(
                torch.from_numpy(nodes_neighbor_times).float().to(self.device)
            )

        nodes_neighbor_time_features = self.projection_layer["time"](
            nodes_neighbor_time_features
        )
        nodes_edge_raw_features = self.projection_layer["edge"](
            self.edge_raw_features[torch.from_numpy(nodes_edge_ids)]
            .to(self.device)[:, :, 0]
            .unsqueeze(-1)
        )

        if self.ablation == "time":
            nodes_neighbor_time_features *= 0
        elif self.ablation == "skill":
            nodes_neighbor_node_raw_features *= 0

        return (
            nodes_neighbor_node_raw_features,
            nodes_edge_raw_features,
            nodes_neighbor_time_features,
        )


class LaychNorm_new(nn.Module):
    def __init__(self, num_channels=2048, epsilon=1e-5, momentum=0.9, rescale=True):
        super().__init__()
        self.num_channels = num_channels
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
        self.eps = epsilon
        self.register_buffer("runningmean", torch.zeros(num_channels))
        self.register_buffer("runningvar", torch.ones(num_channels))

    def forward(self, x):
        std = self.runningvar.add(self.eps).sqrt()
        return x.sub(self.runningmean).div(std)


class LayerNormGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        norm_momentum=0.9,
        norm_epsilon=1e-5,
        use_residual=True,
    ):
        super().__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.norm1 = LaychNorm_new(
            hidden_dim, epsilon=norm_epsilon, momentum=norm_momentum
        )

        self.gru2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.norm2 = LaychNorm_new(
            hidden_dim, epsilon=norm_epsilon, momentum=norm_momentum
        )

        self.use_residual = use_residual

    def forward(self, x, h0=None):
        out, hn1 = self.gru1(x, h0)  # [B,T,D]
        out = self.norm1(out)  # LN
        res = out
        out, hn2 = self.gru2(out, hn1)
        out = self.norm2(out)
        if self.use_residual:
            out = out + res
        return out, hn2


class SpectralEnhancer(nn.Module):
    def __init__(
        self, hidden_dim, freq_gate_dim_ratio=0.25, freq_gate_activation="relu"
    ):
        super().__init__()
        self.freq_gate_dim_ratio = freq_gate_dim_ratio
        self.freq_gate_activation = freq_gate_activation

        if freq_gate_activation.lower() == "relu":
            activation_fn = nn.ReLU()
        elif freq_gate_activation.lower() == "tanh":
            activation_fn = nn.Tanh()
        elif freq_gate_activation.lower() == "sigmoid":
            activation_fn = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation function: {freq_gate_activation}")

        self.freq_gate = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * freq_gate_dim_ratio)),
            activation_fn,
            nn.Linear(int(hidden_dim * freq_gate_dim_ratio), hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape

        x_fft = fft.rfft(x, dim=1)  # [B, T//2+1, D]
        magnitude = torch.abs(x_fft)

        high_freq_idx = int(0.7 * magnitude.size(1))
        high_freq = magnitude[:, high_freq_idx:, :]  # [B, F, D]
        high_freq = torch.mean(high_freq, dim=1)  # [B, D]

        gate = self.freq_gate(high_freq).unsqueeze(1)  # [B, 1, D]

        enhanced = x * gate  # [B, T, D]
        if SAVE_MID_FEATURE:
            gate_save_dir = Path("./mid_feature/gate")
            gate_save_dir.mkdir(parents=True, exist_ok=True)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = gate_save_dir / f"{current_time}.npy"
            np.save(file_path, gate.cpu().numpy())

            enhanced_save_dir = Path("./mid_feature/enhanced")
            enhanced_save_dir.mkdir(parents=True, exist_ok=True)
            current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = enhanced_save_dir / f"{current_time}.npy"
            np.save(file_path, enhanced.cpu().numpy())

        return enhanced


class DSCAKT_Seq(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        node_dim: int,
        ablation: str,
        freq_gate_dim_ratio=0.25,
        freq_gate_activation="relu",
        norm_momentum=0.9,
        norm_epsilon=1e-5,
        use_residual=True,
    ):
        super().__init__()
        self.ablation = ablation

        if self.ablation == "layernormgru":
            self.hid_node_updater = nn.GRU(node_dim, node_dim, batch_first=True)
        else:
            self.hid_node_updater = LayerNormGRU(
                node_dim,
                node_dim,
                norm_momentum=norm_momentum,
                norm_epsilon=norm_epsilon,
                use_residual=use_residual,
            )

        self.spectral_enhancer = SpectralEnhancer(
            node_dim,
            freq_gate_dim_ratio=freq_gate_dim_ratio,
            freq_gate_activation=freq_gate_activation,
        )

    def update(self, x):
        if self.ablation != "spectral":
            x = self.spectral_enhancer(x)
        out, hidden = self.hid_node_updater(x)
        return torch.squeeze(hidden, 0)
