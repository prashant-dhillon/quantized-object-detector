import argparse
import sys, os
sys.path.append(os.getcwd())
import torch.nn as nn
from torch.nn import Module
import torch.nn.utils.prune as prune
import torch


from evaluation.model_evaluation import ModelEvaluator
from utils import helpers



from evaluation import metrics

from Project import project
from models.utils import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from dataloaders.datasets.voc_dataset import VOCDataset


class ModelPruner:
    def __init__(self, amount: float):
        self.amount = amount
        pass

    def __call__(self, model: Module):
        self.__recursive_iterate(model)

    def __prune_layer(self, layer: Module):

        try:

            prune.l1_unstructured(layer, "weight", amount=self.amount)

        except ValueError as e:
            pass

    def __recursive_iterate(self, layer: Module):
        modules = []
        for slayer in layer.modules():
            modules.append(slayer)

        print(f"Before Sparsity {self.__calculate_average_sparsity(modules)}")
        modules = []
        for slayer in layer.modules():
            modules.append(slayer)
            if isinstance(slayer, nn.Conv2d):
                self.__prune_layer(slayer)
        print(f"After Sparsity " f"{self.__calculate_average_sparsity(modules)}")

    def __calculate_average_sparsity(self, modules):

        sum = 0
        considered = 0
        for layer in modules:
            if isinstance(layer, nn.Conv2d):
                if hasattr(layer, "weight") or hasattr(layer, "weight_orig"):
                    considered += 1
                    sum += (
                        100.0
                        * float(torch.sum(layer.weight == 0))
                        / float(layer.weight.nelement())
                    )

        print(f"The number of layers considered is {considered}")
        return sum / considered

    def save_model(self, model):
        file_name = helpers.get_time_of_day() + ".pth"
        torch.save(model.state_dict(), project.pruned_model_dir / file_name)
        return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pruning Script")

    parser.add_argument(
        "--model-path", help="The model file to evaluate", required=True, type=str
    )
    parser.add_argument(
        "--amount", help="Amount of pruning to be applied", required=True, type=float
    )

    args = parser.parse_args()

    model_path = args.model_path

    label_path = project.trained_model_dir / "voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]

    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    net.load(model_path)

    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, nms_method="hard", device=torch.device("cpu")
    )

    dataset = VOCDataset(project.val_data_dir, is_test=True)
    model_evaluator = ModelEvaluator(
        description="Model trained by Navya team with new definition"
    )
    pruner = ModelPruner(amount=args.amount)
    pruner(net)
    pruner.save_model(net)
    model_evaluator.evaluate(
        predictor=predictor,
        dataset=dataset,
        class_names=class_names,
        root_eval_dir=project.eval_results_dir,
        is_pruned=True,
        pruning_percentage=10.0,
        pruning_strategy="20%",
        stopping_point=None,
    )
