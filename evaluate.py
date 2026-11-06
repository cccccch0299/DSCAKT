import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch.nn as nn
import random
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.DyGFormer import DyGFormer
from models.simpleKT import SimpleKT
from models.DyGFormer import DyGFormer
from models.DKT import DKT
from models.DIMKT import DIMKT
from models.IEKT import IEKT
from models.MIKT import MIKT
from models.GIKT import GIKT
from models.AKT import AKT
from models.DTransformer import DTransformer
from models.DyGKT import DyGKT
from models.DSCAKT import DSCAKT
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes
from utils.utils import get_neighbor_sampler, EdgeSampler
from evaluate_models_utils import evaluate_model_link_classification
from utils.DataLoader import (
    get_idx_data_loader,
    get_link_classification_data,
    load_and_map_graph,
)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_classification_args

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from tqdm import tqdm

PLOT = False

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # get arguments
    args = get_link_classification_args(is_evaluation=True)

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

    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    if args.negative_sample_strategy != "random":
        val_neg_edge_sampler = EdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            interact_times=full_data.node_interact_times,
            last_observed_time=train_data.node_interact_times[-1],
            negative_sample_strategy=args.negative_sample_strategy,
            seed=0,
        )
        new_node_val_neg_edge_sampler = EdgeSampler(
            src_node_ids=new_node_val_data.src_node_ids,
            dst_node_ids=new_node_val_data.dst_node_ids,
            interact_times=new_node_val_data.node_interact_times,
            last_observed_time=train_data.node_interact_times[-1],
            negative_sample_strategy=args.negative_sample_strategy,
            seed=1,
        )
        test_neg_edge_sampler = EdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            interact_times=full_data.node_interact_times,
            last_observed_time=val_data.node_interact_times[-1],
            negative_sample_strategy=args.negative_sample_strategy,
            seed=2,
        )
        new_node_test_neg_edge_sampler = EdgeSampler(
            src_node_ids=new_node_test_data.src_node_ids,
            dst_node_ids=new_node_test_data.dst_node_ids,
            interact_times=new_node_test_data.node_interact_times,
            last_observed_time=val_data.node_interact_times[-1],
            negative_sample_strategy=args.negative_sample_strategy,
            seed=3,
        )
    else:
        val_neg_edge_sampler = EdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=0,
        )
        new_node_val_neg_edge_sampler = EdgeSampler(
            src_node_ids=new_node_val_data.src_node_ids,
            dst_node_ids=new_node_val_data.dst_node_ids,
            seed=1,
        )
        test_neg_edge_sampler = EdgeSampler(
            src_node_ids=full_data.src_node_ids,
            dst_node_ids=full_data.dst_node_ids,
            seed=2,
        )
        new_node_test_neg_edge_sampler = EdgeSampler(
            src_node_ids=new_node_test_data.src_node_ids,
            dst_node_ids=new_node_test_data.dst_node_ids,
            seed=3,
        )

    # get data loaders
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
        args.load_model_name = f"{args.model_name}_seed{args.seed}"
        args.save_result_name = f"{args.negative_sample_strategy}_negative_sampling_{args.model_name}_seed{args.seed}"

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/",
            exist_ok=True,
        )
        inf = "ablation paper"
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_result_name}/{inf + str(time.time())}.log"
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

        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f"configuration is {args}")

        if args.model_name == "DKT":
            dynamic_backbone = DKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DyGKT":
            dynamic_backbone = DyGKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
                ablation=args.ablation,
            )
        elif args.model_name == "DSCAKT":
            dynamic_backbone = DSCAKT(
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
            dynamic_backbone = AKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DIMKT":
            dynamic_backbone = DIMKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                dataset_name=args.dataset_name,
                device=args.device,
            )
        elif args.model_name == "DTransformer":
            dynamic_encoder = DTransformer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
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
            dynamic_backbone = IEKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "MIKT":
            dynamic_backbone = MIKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                device=args.device,
            )

        elif args.model_name == "simpleKT":
            dynamic_backbone = SimpleKT(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                dropout=args.dropout,
                num_neighbors=args.num_neighbors,
                device=args.device,
            )
        elif args.model_name == "DyGFormer":
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=full_neighbor_sampler,
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
        link_predictor = MergeLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            output_dim=1,
        )
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f"model -> {model}")
        logger.info(
            f"model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, "
            f"{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB."
        )

        # load the saved model
        load_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.load_model_name}"
        early_stopping = EarlyStopping(
            patience=0,
            save_model_folder=load_model_folder,
            save_model_name=args.load_model_name,
            logger=logger,
            model_name=args.model_name,
        )
        early_stopping.load_checkpoint(model, map_location="cpu")

        model = convert_to_gpu(model, device=args.device)
        # put the node raw messages of memory-based models on device

        loss_func = nn.BCELoss()

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

        if PLOT:
            student_id = random.choice(val_data.src_node_ids)
            student_indices = [
                i
                for i, src_id in enumerate(val_data.src_node_ids)
                if src_id == student_id
            ]
            student_indices.sort(key=lambda i: val_data.node_interact_times[i])
            num_samples = min(15, len(student_indices))
            sample_indices = student_indices[:num_samples]
            demo_seq = [
                (
                    val_data.src_node_ids[i],
                    val_data.dst_node_ids[i],
                    val_data.edge_ids[i],
                    val_data.node_interact_times[i],
                    val_data.labels[i],
                )
                for i in sample_indices
            ]
            src_ids = np.array([x[0] for x in demo_seq])
            dst_ids = np.array([x[1] for x in demo_seq])
            edge_ids = np.array([x[2] for x in demo_seq])
            times = np.array([x[3] for x in demo_seq])
            labels = torch.tensor([x[4] for x in demo_seq], dtype=torch.float32)

            mastery_scores = []
            pred_probs = []
            log_delta_t = []

            with torch.no_grad():
                for step in tqdm(range(len(demo_seq)), desc="infer"):
                    s, d, e, t = (
                        src_ids[step : step + 1],
                        dst_ids[step : step + 1],
                        edge_ids[step : step + 1],
                        times[step : step + 1],
                    )
                    src_emb, dst_emb = (
                        dynamic_backbone.compute_embeddings(
                            s, e, t, d
                        )
                    )
                    mastery = src_emb.mean(dim=-1).item()
                    logit = (src_emb * dst_emb).sum(dim=-1)
                    prob = torch.sigmoid(logit).item()
                    mastery_scores.append(mastery * 1000)
                    pred_probs.append(prob)

                    if step > 0:
                        log_delta_t.append(np.log(times[step] - times[step - 1]))
                    else:
                        log_delta_t.append(0)

            time_intervals = np.diff(times, prepend=times[0])
            min_val = np.min(log_delta_t)
            max_val = np.max(log_delta_t)
            bins = np.linspace(min_val, max_val, 5)  # 4 段需要 5 个边界值
            cmap = mcolors.ListedColormap(["#e0f3f8", "#abd9e9", "#4575b4", "#313695"])

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.plot(
                range(len(demo_seq)),
                mastery_scores,
                color="black",
                lw=2,
            )

            sc = ax.scatter(
                range(len(demo_seq)),
                log_delta_t,
                c=log_delta_t,
                cmap="gray",
                zorder=1,
                visible=False,
            )

            cbar = plt.colorbar(sc, ax=ax, orientation="vertical")
            cbar.set_label("log Δt_i")

            ax.set_xlim(0, len(demo_seq))
            ax.set_xlabel("Step")
            ax.set_ylabel("Mastery degree")
            ax.set_title(
                f"Mastery estimation of student {student_id} during the process of solving problems"
            )

            ax4 = ax.twiny()
            ax4.set_xlim(ax.get_xlim())
            ax4.set_xticks(range(len(demo_seq)))
            ax4.set_xticklabels([])
            for i, iv in enumerate(log_delta_t):
                color = cmap(np.digitize(iv, bins) - 1)
                ax4.axvspan(i, i + 1, color=color, alpha=0.6)

            ax2 = ax.twinx()
            ax2.set_ylim(-0.1, 1.1)
            ax2.set_yticks([])
            ax2.set_xticks(range(len(demo_seq)))
            for i, label in enumerate(labels):
                if label == 1:
                    ax2.text(
                        i, 0, "✓", fontsize=14, color="green", ha="center", va="center"
                    )
                else:
                    ax2.text(
                        i, 0, "✗", fontsize=14, color="red", ha="center", va="center"
                    )

            plt.savefig("mastery_vis.png")

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
            save_result_folder, f"{args.save_result_name}.json"
        )
        with open(save_result_path, "w") as file:
            file.write(result_json)
        logger.info(f"save negative sampling results at {save_result_path}")

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

        # sys.exit()
