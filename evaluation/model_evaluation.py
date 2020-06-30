import torch
import numpy as np
import sys, os
import argparse

sys.path.append(os.getcwd())

from evaluation import metrics

from Project import project
from models.utils import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from dataloaders.datasets.voc_dataset import VOCDataset
from tqdm import tqdm
import pandas as pd
from utils.helpers import get_time_of_day
import time


class ModelEvaluator:
    def __init__(self, description):
        self.description = description
        pass

    def _evaluate_dataset(self, predictor, dataset, stopping_point):
        results = []
        inference_times = []
        for i in tqdm(range(len(dataset))):
            if i == stopping_point:
                break
            image = dataset.get_image(i)
            start = time.time()
            boxes, labels, probs = predictor.predict(image)
            end = time.time()
            inference_times.append(end - start)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            results.append(
                torch.cat(
                    [
                        indexes.reshape(-1, 1),
                        labels.reshape(-1, 1).float(),
                        probs.reshape(-1, 1),
                        boxes + 1.0,  # matlab's indexes start from 1
                    ],
                    dim=1,
                )
            )

        return (
            results,
            np.mean(np.array(inference_times)),
            np.std(np.array(inference_times)),
        )

    def _generate_prediction_files(self, class_names, eval_dir, results, dataset):
        for class_index, class_name in enumerate(class_names):
            if class_index == 0:
                continue  # ignore background
            prediction_path = eval_dir / f"det_test_{class_name}.txt"
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    image_id = dataset.ids[int(sub[i, 0])]
                    print(image_id + " " + " ".join([str(v) for v in prob_box]), file=f)

    def _generate_csv_for_metrics(self, metric_dict, eval_dir):

        df = pd.DataFrame(metric_dict.items())
        df.to_csv(eval_dir / "metrics.csv")

    def _get_mAP(self, class_names, eval_dir, iou_threshold, dataset):
        class_name_precision = dict()
        aps = []
        (
            true_case_stat,
            all_gb_boxes,
            all_difficult_cases,
        ) = metrics.group_annotation_by_class(dataset)
        for class_index, class_name in enumerate(class_names):
            prediction_path = eval_dir / f"det_test_{class_name}.txt"
            if class_index == 0:
                continue
            # prediction_path = eval_path / f"det_test_{class_name}.txt"
            ap = metrics.compute_average_precision_per_class(
                true_case_stat[class_index],
                all_gb_boxes[class_index],
                all_difficult_cases[class_index],
                prediction_path,
                iou_threshold,
                False,
            )
            class_name_precision[class_name] = ap
            aps.append(ap)
        class_name_precision["mean_average_precision"] = sum(aps) / len(aps)
        return class_name_precision

    def evaluate(
        self,
        predictor,
        dataset,
        class_names,
        root_eval_dir,
        iou_threshold=0.5,
        is_pruned=False,
        pruning_strategy="",
        pruning_percentage: float = 0.0,
        stopping_point=None,
    ):
        stopping_point = stopping_point if stopping_point is not None else -1
        sparsity = metrics.calculate_average_sparsity(predictor.net)
        if int(sparsity) > 2 and not is_pruned:
            raise ValueError(
                "The Sparsity shows that this model is pruned but you didn't get the prune config"
            )
        root_eval_dir.mkdir(exist_ok=True)
        eval_dir = root_eval_dir / get_time_of_day()
        eval_dir.mkdir(exist_ok=True)

        results, mean_inference_time, std_inference_time = self._evaluate_dataset(
            predictor, dataset, stopping_point
        )
        results = torch.cat(results)

        self._generate_prediction_files(
            class_names=class_names, dataset=dataset, eval_dir=eval_dir, results=results
        )
        class_name_precision = self._get_mAP(
            class_names=class_names,
            eval_dir=eval_dir,
            iou_threshold=iou_threshold,
            dataset=dataset,
        )

        metric_dict = class_name_precision
        metric_dict["title"] = self.description
        metric_dict["sparsity"] = sparsity
        metric_dict["macs"] = metrics.get_macs(predictor.net)
        metric_dict["mem_usage"] = str(metrics.get_memory_used()) + "MB"
        metric_dict["model_size"] = str(metrics.get_size_of_model(predictor.net)) + "MB"
        metric_dict["is_pruned"] = str(is_pruned)
        metric_dict["pruning_strategy"] = pruning_strategy
        metric_dict["pruning_percentage"] = pruning_percentage
        metric_dict["mean_inference_time"] = mean_inference_time
        metric_dict["std_inference_time"] = std_inference_time
        self._generate_csv_for_metrics(metric_dict, eval_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation Script")

    parser.add_argument(
        "--model-path", help="The model file to evaluate", required=True, type=str
    )
    args = parser.parse_args()


    model_path = args.model_path

    label_path = project.trained_model_dir / "voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]

    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True, 
            device=torch.device("cpu")
            )
    net.load(model_path)

    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, nms_method="hard", device=torch.device("cpu")
    )

    dataset = VOCDataset(project.val_data_dir, is_test=True)
    model_evaluator = ModelEvaluator(
        description="Model trained by Navya team with new definition"
    )

    model_evaluator.evaluate(
        predictor=predictor,
        dataset=dataset,
        class_names=class_names,
        root_eval_dir=project.eval_results_dir,
        stopping_point=None,
    )
