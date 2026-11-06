import logging
import time
import sys
import os
from tqdm import tqdm
import time
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.DyGKT import DyGKT
from models.DSCAKT import DSCAKT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.simpleKT import SimpleKT
from models.DyGFormer import DyGFormer
from models.DKT import DKT
from models.DIMKT import DIMKT
from models.IEKT import IEKT
from models.AKT import AKT
from models.MIKT import MIKT
from models.GIKT import GIKT
from models.DTransformer import DTransformer
from models.modules import MergeLayer, MLPClassifier
from utils.utils import (
    set_random_seed,
    convert_to_gpu,
    get_parameter_sizes,
    create_optimizer,
)
from utils.utils import get_neighbor_sampler, EdgeSampler
from evaluate_models_utils import evaluate_model_link_classification
from utils.metrics import get_link_classification_metrics
from utils.DataLoader import (
    get_idx_data_loader,
    get_link_classification_data,
    load_and_map_graph,
)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_classification_args
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import sys
import io


sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")


class LinkLoss(nn.Module):

    def __init__(
        self,
        alpha=0.7,
        pos_weight_fn=True,
        distill_weight=0.3,
        ema_gamma=0.9,
        max_buffer=50000,
    ):
        super().__init__()
        self.alpha = alpha
        self.pos_weight_fn = pos_weight_fn
        self.distill_weight = distill_weight
        self.ema_gamma = ema_gamma
        self.register_buffer("pos_total", torch.tensor(0.0))
        self.register_buffer("neg_total", torch.tensor(0.0))
        self.pred_buffer = deque(maxlen=max_buffer)
        self.idx_map = {}

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        timestamps: torch.Tensor = None,
        indices: torch.Tensor = None,
    ):
        device = input.device
        loss_bce = F.binary_cross_entropy(input, target, reduction="none")

        if self.pos_weight_fn:
            with torch.no_grad():
                self.pos_total += target.sum()
                self.neg_total += (1 - target).sum()
                pos_w = (self.neg_total + 1) / (self.pos_total + 1)
            loss_bce = torch.where(target == 1, loss_bce * pos_w, loss_bce)

        if timestamps is not None:
            t_norm = (timestamps - timestamps.min()) / (
                timestamps.max() - timestamps.min() + 1e-8
            )
            time_w = torch.exp(self.alpha * t_norm)
            loss_bce = loss_bce * time_w

        if indices is not None and len(self.pred_buffer) > 0:
            ema_preds = []
            valid_mask = []
            for i, idx in enumerate(indices.cpu().tolist()):
                pos = self.idx_map.get(idx, None)
                if pos is not None:
                    ema_preds.append(self.pred_buffer[pos][1])
                    valid_mask.append(i)
            if len(ema_preds) > 0:
                ema_preds = torch.tensor(ema_preds, device=device)
                valid_mask = torch.tensor(valid_mask, device=device)
                student = input[valid_mask]
                loss_distill = F.mse_loss(student, ema_preds, reduction="none")
                conf_mask = (ema_preds > 0.65) | (ema_preds < 0.35)
                if conf_mask.any():
                    loss_distill = loss_distill[conf_mask].mean()
                else:
                    loss_distill = 0.0
            else:
                loss_distill = 0.0
        else:
            loss_distill = 0.0

        if indices is not None:
            with torch.no_grad():
                for i, idx in enumerate(indices.cpu().tolist()):
                    ema_p = input[i].item()
                    pos = self.idx_map.get(idx, None)
                    if pos is not None:  # 已存在，更新 EMA
                        old_ema = self.pred_buffer[pos][1]
                        new_ema = (
                            self.ema_gamma * old_ema + (1 - self.ema_gamma) * ema_p
                        )
                        self.pred_buffer[pos] = (idx, new_ema)
                    else:
                        if len(self.pred_buffer) == self.pred_buffer.maxlen:
                            old_idx, _ = self.pred_buffer.popleft()
                            del self.idx_map[old_idx]
                        self.pred_buffer.append((idx, ema_p))
                        self.idx_map[idx] = len(self.pred_buffer) - 1

        total = loss_bce.mean()
        if isinstance(loss_distill, torch.Tensor):
            total = total + self.distill_weight * loss_distill
        return total


if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    torch.autograd.set_detect_anomaly(True)

    # get arguments
    args = get_link_classification_args(is_evaluation=False)
    torch.save(args, "args.pt")

    # get data for training, validation and testing
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
    # print(node_raw_features)
    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = EdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids
    )
    val_neg_edge_sampler = EdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0
    )
    new_node_val_neg_edge_sampler = EdgeSampler(
        src_node_ids=new_node_val_data.src_node_ids,
        dst_node_ids=new_node_val_data.dst_node_ids,
        seed=1,
    )
    test_neg_edge_sampler = EdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2
    )
    new_node_test_neg_edge_sampler = EdgeSampler(
        src_node_ids=new_node_test_data.src_node_ids,
        dst_node_ids=new_node_test_data.dst_node_ids,
        seed=3,
    )

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_val_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    new_node_test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(new_node_test_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )

    (
        val_metric_all_runs,
        new_node_val_metric_all_runs,
        test_metric_all_runs,
        new_node_test_metric_all_runs,
    ) = ([], [], [], [])

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f"{args.model_name}_seed{args.seed}"

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
            exist_ok=True,
        )
        inf = "paper"
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{inf+str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        start_time = time.time()
        initial_memory = torch.cuda.memory_allocated()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f"configuration is {args}")

        # create model

        if args.model_name == "DKT":
            dynamic_encoder = DKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DyGKT":
            dynamic_encoder = DyGKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
                ablation=args.ablation,
            )
        elif args.model_name == "DSCAKT":
            dynamic_encoder = DSCAKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                knowledge_graph_edge_index=kg_edge_index,  # .to(device),
                knowledge_graph_edge_type=kg_edge_type,  # .to(device),
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
        elif args.model_name == "AKT":
            dynamic_encoder = AKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DIMKT":
            dynamic_encoder = DIMKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                dataset_name=args.dataset_name,
                device=args.device,
            )
        elif args.model_name == 'GIKT':
            dynamic_backbone = GIKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout, 
                num_neighbors=args.num_neighbors,
                device=args.device
            )
        elif args.model_name == "IEKT":
            dynamic_encoder = IEKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "DTransformer":
            dynamic_encoder = DTransformer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "MIKT":
            dynamic_encoder = MIKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                device=args.device,
            )

        elif args.model_name == "simpleKT":
            dynamic_encoder = SimpleKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DyGFormer":
            dynamic_encoder = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        if args.model_name == "DyGKT" or args.model_name == "DSCAKT":
            link_predictor = MergeLayer(
                input_dim1=64, input_dim2=64, hidden_dim=64, output_dim=1
            ).to(args.device)
        else:
            link_predictor = MergeLayer(
                node_raw_features.shape[1],
                input_dim2=node_raw_features.shape[1],
                hidden_dim=node_raw_features.shape[1],
                output_dim=1,
            ).to(
                args.device
            )  # input_dim1=64, input_dim2=64,hidden_dim=64,output_dim=1)#node_raw_features.shape[1],input_dim2=node_raw_features.shape[1], hidden_dim=node_raw_features.shape[1], output_dim=1)
        # MLPClassifier(node_dim, dropout=args.dropout)
        model = nn.Sequential(dynamic_encoder, link_predictor)
        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
            f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
        )

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        # loss_func = LinkLoss(alpha=0.7, distill_weight=0.3).to(args.device)
        loss_func = nn.BCELoss()
        torch.autograd.set_detect_anomaly(True)
        final_memory = None

        for epoch in range(args.num_epochs):
            if args.test and epoch > 0:
                end_time = time.time()
                memory_used = final_memory - initial_memory
                logger.info(f"USE_TIME: {end_time-start_time}")
                logger.info(f"USE_GPU: {memory_used / (1024**2):.3f} MB")
                logger.info(
                    f"MODEL_PARA:{get_parameter_sizes(model) * 4 / 1024 / 1024:.3f} MB"
                )
                print(args.model_name)
                sys.exit()
            model.train()
            if args.model_name in [
                "DyGKT",
                "MIKT",
                "DTransformer",
                "IEKT",
                "DIMKT",
                "DyGFormer",
                "DKT",
                "AKT",
                "GIKT",
                "simpleKT",
                "DSCAKT",
            ]:
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            train_losses, train_metrics = [], []
            train_predicts, train_labels = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                    batch_edge_labels,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                    train_data.labels[train_data_indices],
                )

                if args.model_name in [
                    "DyGKT",
                    "DKT",
                    "AKT",
                    "DTransformer",
                    "MIKT",
                    "GIKT"
                    "simpleKT",
                    "DSCAKT",
                ]:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_embeddings(
                        src_node_ids=batch_src_node_ids,
                        edge_ids=batch_edge_ids,
                        node_interact_times=batch_node_interact_times,
                        dst_node_ids=batch_dst_node_ids,
                    )

                elif args.model_name in ["MIKT", "IEKT", "DIMKT"]:
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        edge_ids=batch_edge_ids,
                    )

                elif args.model_name in ["DyGFormer"]:
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
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")

                predicts = (
                    model[1](batch_src_node_embeddings, batch_dst_node_embeddings)
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                labels = torch.tensor(
                    batch_edge_labels, dtype=torch.float32, device=args.device
                )
                loss = loss_func(input=predicts, target=labels)
                final_memory = torch.cuda.memory_allocated()
                train_losses.append(loss.item())

                train_predicts.append(predicts)
                train_labels.append(labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f"Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}"
                )

            val_losses, val_metrics = evaluate_model_link_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

            new_node_val_losses, new_node_val_metrics = (
                evaluate_model_link_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=new_node_val_idx_data_loader,
                    evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                    evaluate_data=new_node_val_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )
            )

            train_predict = torch.cat(train_predicts, dim=0)
            train_label = torch.cat(train_labels, dim=0)
            # standard_label = torch.tensor([0, 1], dtype=torch.float32, device=args.device)
            # predicts,labels = torch.concat((predicts,standard_label)),torch.concat((labels,standard_label))

            train_metrics.append(
                get_link_classification_metrics(
                    predicts=train_predict, labels=train_label
                )
            )

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}'
            )
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f"train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}"
                )
            logger.info(f"validate loss: {np.mean(val_losses):.4f}")
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f"validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}"
                )
            logger.info(f"new node validate loss: {np.mean(new_node_val_losses):.4f}")
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(
                    f"new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}"
                )

            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_classification(
                    model_name=args.model_name,
                    model=model,
                    neighbor_sampler=full_neighbor_sampler,
                    evaluate_idx_data_loader=test_idx_data_loader,
                    evaluate_neg_edge_sampler=test_neg_edge_sampler,
                    evaluate_data=test_data,
                    loss_func=loss_func,
                    num_neighbors=args.num_neighbors,
                    time_gap=args.time_gap,
                )

                new_node_test_losses, new_node_test_metrics = (
                    evaluate_model_link_classification(
                        model_name=args.model_name,
                        model=model,
                        neighbor_sampler=full_neighbor_sampler,
                        evaluate_idx_data_loader=new_node_test_idx_data_loader,
                        evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                        evaluate_data=new_node_test_data,
                        loss_func=loss_func,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )
                )

                logger.info(f"test loss: {np.mean(test_losses):.4f}")
                for metric_name in test_metrics[0].keys():
                    logger.info(
                        f"test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}"
                    )
                logger.info(f"new node test loss: {np.mean(new_node_test_losses):.4f}")
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(
                        f"new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}"
                    )

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (
                        metric_name,
                        np.mean(
                            [val_metric[metric_name] for val_metric in val_metrics]
                        ),
                        True,
                    )
                )
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f"get final performance on dataset {args.dataset_name}...")

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        val_losses, val_metrics = evaluate_model_link_classification(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=val_idx_data_loader,
            evaluate_neg_edge_sampler=val_neg_edge_sampler,
            evaluate_data=val_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        new_node_val_losses, new_node_val_metrics = evaluate_model_link_classification(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=new_node_val_idx_data_loader,
            evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
            evaluate_data=new_node_val_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        test_losses, test_metrics = evaluate_model_link_classification(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        new_node_test_losses, new_node_test_metrics = (
            evaluate_model_link_classification(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=new_node_test_idx_data_loader,
                evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                evaluate_data=new_node_test_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )
        )
        # store the evaluation metrics at the current run
        (
            val_metric_dict,
            new_node_val_metric_dict,
            test_metric_dict,
            new_node_test_metric_dict,
        ) = ({}, {}, {}, {})

        logger.info(f"validate loss: {np.mean(val_losses):.4f}")
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean(
                [val_metric[metric_name] for val_metric in val_metrics]
            )
            logger.info(f"validate {metric_name}, {average_val_metric:.4f}")
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f"new node validate loss: {np.mean(new_node_val_losses):.4f}")
        for metric_name in new_node_val_metrics[0].keys():
            average_new_node_val_metric = np.mean(
                [
                    new_node_val_metric[metric_name]
                    for new_node_val_metric in new_node_val_metrics
                ]
            )
            logger.info(
                f"new node validate {metric_name}, {average_new_node_val_metric:.4f}"
            )
            new_node_val_metric_dict[metric_name] = average_new_node_val_metric

        logger.info(f"test loss: {np.mean(test_losses):.4f}")
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean(
                [test_metric[metric_name] for test_metric in test_metrics]
            )
            logger.info(f"test {metric_name}, {average_test_metric:.4f}")
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f"new node test loss: {np.mean(new_node_test_losses):.4f}")
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean(
                [
                    new_node_test_metric[metric_name]
                    for new_node_test_metric in new_node_test_metrics
                ]
            )
            logger.info(
                f"new node test {metric_name}, {average_new_node_test_metric:.4f}"
            )
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f"Run {run + 1} cost {single_run_time:.2f} seconds.")

        val_metric_all_runs.append(val_metric_dict)
        new_node_val_metric_all_runs.append(new_node_val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result

        result_json = {
            "validate metrics": {
                metric_name: f"{val_metric_dict[metric_name]:.4f}"
                for metric_name in val_metric_dict
            },
            "new node validate metrics": {
                metric_name: f"{new_node_val_metric_dict[metric_name]:.4f}"
                for metric_name in new_node_val_metric_dict
            },
            "test metrics": {
                metric_name: f"{test_metric_dict[metric_name]:.4f}"
                for metric_name in test_metric_dict
            },
            "new node test metrics": {
                metric_name: f"{new_node_test_metric_dict[metric_name]:.4f}"
                for metric_name in new_node_test_metric_dict
            },
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(
            save_result_folder, f"{args.save_model_name}.json"
        )

        with open(save_result_path, "w") as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f"metrics over {args.num_runs} runs:")

    for metric_name in val_metric_all_runs[0].keys():
        logger.info(
            f"validate {metric_name}, {[val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]}"
        )
        logger.info(
            f"average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} "
            f"± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}"
        )

    for metric_name in new_node_val_metric_all_runs[0].keys():
        logger.info(
            f"new node validate {metric_name}, {[new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]}"
        )
        logger.info(
            f"average new node validate {metric_name}, {np.mean([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs]):.4f} "
            f"± {np.std([new_node_val_metric_single_run[metric_name] for new_node_val_metric_single_run in new_node_val_metric_all_runs], ddof=1):.4f}"
        )

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f"test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}"
        )
        logger.info(
            f"average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} "
            f"± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}"
        )

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(
            f"new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}"
        )
        logger.info(
            f"average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} "
            f"± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}"
        )

    sys.exit()
