import numpy as np
import torch
import logging
import os
import time
import json
from tqdm import tqdm
from collections import defaultdict, Counter
from models.DSCAKT import DSCAKT as DSCAKT
from utils.DataLoader import load_and_map_graph
from evaluate import *
from evaluate_models_utils import evaluate_model_link_classification
from utils.metrics import get_link_classification_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_cold_start_indices(eval_data, max_k=5):
    q_seen = Counter()
    cold_indices = defaultdict(list)
    for idx, q in tqdm(enumerate(eval_data.dst_node_ids), desc="counting q"):
        k = q_seen[q] + 1
        if k <= max_k:
            cold_indices[k].append(idx)
        q_seen[q] += 1
    return cold_indices


def evaluate_cold_start_metrics(
    model_name: str,
    model: torch.nn.Module,
    neighbor_sampler,
    eval_data,
    loss_func,
    num_neighbors: int = 50,
    time_gap: int = 2000,
    max_k: int = 5,
    batch_size: int = 2048,
):
    cold_indices = get_cold_start_indices(eval_data, max_k=max_k)
    results = []
    for k in range(1, max_k + 1):
        if len(cold_indices[k]) == 0:
            logger.warning(f"k={k} no sample,skip")
            continue

        from utils.DataLoader import get_idx_data_loader

        k_loader = get_idx_data_loader(
            indices_list=cold_indices[k],
            batch_size=batch_size,
            shuffle=False,
        )

        losses, metrics = evaluate_model_link_classification(
            model_name=model_name,
            model=model,
            neighbor_sampler=neighbor_sampler,
            evaluate_idx_data_loader=k_loader,
            evaluate_neg_edge_sampler=EdgeSampler(
                src_node_ids=eval_data.src_node_ids,
                dst_node_ids=eval_data.dst_node_ids,
                seed=42,
            ),
            evaluate_data=eval_data,
            loss_func=loss_func,
            num_neighbors=num_neighbors,
            time_gap=time_gap,
        )

        auc = np.mean([m["roc_auc"] for m in metrics])
        prec = np.mean([m["average_precision"] for m in metrics])
        results.append(
            {"k": k, "auc": auc, "precision": prec, "n_sample": len(cold_indices[k])}
        )
    return results


if __name__ == "__main__":

    args = get_link_classification_args(is_evaluation=True)
    NODE_FILE = f"./processed_data/{args.dataset_name}/node.csv"
    EDGE_FILE = f"./processed_data/{args.dataset_name}/edge.csv"
    try:
        kg_edge_index, kg_edge_type, num_kg_nodes, num_kg_relations, kg_id_map = (
            load_and_map_graph(NODE_FILE, EDGE_FILE)
        )
    except FileNotFoundError:
        print("Knowledge graph files not found, running original DyGKT model.")
        use_kg = False
    else:
        use_kg = True
    print(use_kg)

    (
        node_raw_features,
        edge_raw_features,
        full_data,
        train_data,
        val_data,
        test_data,
        new_node_val_data,
        new_node_test_data,
    ) = get_link_classification_data(
        dataset_name=args.dataset_name,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    if args.model_name == "DSCAKT":

        dynamic_backbone = DSCAKT(
            node_raw_features=node_raw_features,
            edge_raw_features=edge_raw_features,
            knowledge_graph_edge_index=kg_edge_index,
            knowledge_graph_edge_type=kg_edge_type,
            num_kg_nodes=num_kg_nodes,
            num_kg_relations=num_kg_relations,
            kg_id_map=kg_id_map,
            dropout=args.dropout,
            num_neighbors=args.num_neighbors,
            device=args.device,
            ablation=args.ablation,
            freq_gate_dim_ratio=args.freq_gate_dim_ratio,
            freq_gate_activation=args.freq_gate_activation,
            norm_momentum=args.norm_momentum,
            norm_epsilon=args.norm_epsilon,
        )
        link_predictor = MergeLayer(
            input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1
        )
        model = nn.Sequential(dynamic_backbone, link_predictor)
        model = convert_to_gpu(model, device=args.device)

        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.model_name}_seed0"

        early_stopping = EarlyStopping(
            patience=0,
            save_model_folder=load_model_folder,
            save_model_name=f"{args.model_name}_seed0",
            logger=logging.getLogger(),
            model_name=args.model_name,
        )
        early_stopping.load_checkpoint(model, map_location="cpu")
    else:
        raise ValueError("error")

    loss_func = nn.BCELoss()

    cold_results = evaluate_cold_start_metrics(
        model_name=args.model_name,
        model=model,
        neighbor_sampler=full_neighbor_sampler,
        eval_data=test_data,
        loss_func=loss_func,
        num_neighbors=args.num_neighbors,
        time_gap=args.time_gap,
        max_k=15,
        batch_size=args.batch_size,
    )

    logger.info("==== Cold-Start Performance ====")
    for res in cold_results:
        logger.info(
            f"The new question appears for the {res['k']} time："
            f"AUC={res['auc']:.4f}, "
            f"Precision={res['precision']:.4f}, "
            f"sample={res['n_sample']}"
        )

    save_path = f"./saved_results/cold_start_{args.model_name}_{args.dataset_name}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(cold_results, f, indent=4)
    logger.info(f"result saved in {save_path}")
