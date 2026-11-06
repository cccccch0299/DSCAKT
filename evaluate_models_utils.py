import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from utils.metrics import get_link_classification_metrics
from utils.utils import EdgeSampler, NeighborSampler
from utils.DataLoader import Data


def evaluate_model_link_classification(
    model_name: str,
    model: nn.Module,
    neighbor_sampler: NeighborSampler,
    evaluate_idx_data_loader: DataLoader,
    evaluate_neg_edge_sampler: EdgeSampler,
    evaluate_data: Data,
    loss_func: nn.Module,
    num_neighbors: int = 20,
    time_gap: int = 2000,
):
    """
    evaluate models on the link classification task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: EdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)

    model.eval()

    total_inference_time = 0.0
    total_sample_count = 0
    with torch.no_grad():
        evaluate_losses, evaluate_metrics = [], []
        evaluate_predicts, evaluate_labels = [], []
        evaluate_idx_data_loader_tqdm = tqdm(evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(
            evaluate_idx_data_loader_tqdm
        ):
            evaluate_data_indices = evaluate_data_indices.numpy()
            (
                batch_src_node_ids,
                batch_dst_node_ids,
                batch_node_interact_times,
                batch_edge_ids,
                batch_edge_labels,
            ) = (
                evaluate_data.src_node_ids[evaluate_data_indices],
                evaluate_data.dst_node_ids[evaluate_data_indices],
                evaluate_data.node_interact_times[evaluate_data_indices],
                evaluate_data.edge_ids[evaluate_data_indices],
                evaluate_data.labels[evaluate_data_indices],
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            if model_name in [
                "DyGKT",
                "DSCAKT",
                "IEKT",
                "DKT",
                "AKT",
                "simpleKT",
            ]:

                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_embeddings(
                    src_node_ids=batch_src_node_ids,
                    edge_ids=batch_edge_ids,
                    node_interact_times=batch_node_interact_times,
                    dst_node_ids=batch_dst_node_ids,
                )

            elif model_name in ["DyGFormer"]:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = model[
                    0
                ].compute_embeddings(
                    src_node_ids=batch_src_node_ids,
                    dst_node_ids=batch_dst_node_ids,
                    node_interact_times=batch_node_interact_times,
                )

            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            predicts = (
                model[1](batch_src_node_embeddings, batch_dst_node_embeddings)
                .squeeze(dim=-1)
                .sigmoid()
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()

            total_inference_time += end_time - start_time
            total_sample_count += len(batch_src_node_ids)

            labels = torch.tensor(
                batch_edge_labels, dtype=torch.float32, device=predicts.device
            )

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())

            evaluate_predicts.append(predicts)
            evaluate_labels.append(labels)

            evaluate_idx_data_loader_tqdm.set_description(
                f"evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}"
            )

        # if total_sample_count > 0:
        #     print(f"[{model_name}] Total Inference Time: {total_inference_time:.4f}s")
        #     print(
        #         f"[{model_name}] Average Latency per Sample: {total_inference_time / total_sample_count * 1000:.4f} ms"
        #     )
        #     print(
        #         f"[{model_name}] Throughput: {total_sample_count / total_inference_time:.2f} samples/s"
        #     )

        evaluate_predict = torch.cat(evaluate_predicts, dim=0)
        evaluate_label = torch.cat(evaluate_labels, dim=0)

        evaluate_metrics.append(
            get_link_classification_metrics(
                predicts=evaluate_predict, labels=evaluate_label
            )
        )

    return evaluate_losses, evaluate_metrics
