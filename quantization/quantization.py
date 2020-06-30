import argparse
import os
import sys

from tqdm import tqdm

sys.path.append(os.getcwd())

from io import open
import torch
from dataloaders.datasets.voc_dataset import VOCDataset
import torch.quantization
import torch.quantization
from utils import helpers
from Project import project

from models.utils import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from torch.quantization import (
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver,
    QConfig,
    default_per_channel_weight_observer,
)
from models import mobilenet_v2
from evaluation import model_evaluation


class ModelQuantizer:
    def __init__(self):
        pass

    def quantize_def(self, model):
        model.eval().to("cpu")
        model.fuse_model()
        model.qconfig = torch.quantization.default_per_channel_qconfig
        torch.quantization.prepare(model, inplace=True)
        torch.quantization.convert(model, inplace=True)

    def quantize(self, model: mobilenet_v2.MobileNetV2):
        model.eval().to("cpu")

        model.fuse_model()
        # default_per_channel_weight_observer

        model.qconfig = torch.quantization.default_per_channel_qconfig
        #     QConfig(
        #     activation=PerChannelMinMaxObserver.with_args(dtype=torch.qint8),
        #     weight=PerChannelMinMaxObserver.with_args(
        #         dtype=torch.qint8,
        #         qscheme=torch.per_channel_symmetric
        #     )
        # )
        # model.qconfig = torch.quantization.get_default_qconfig("fbgemm")

        torch.quantization.prepare(model, inplace=True)
        # We should calibrate here
        print("Calibrating")
        self._calibrate(model, 10)
        print("Done Calibrating")
        torch.quantization.convert(model, inplace=True)

    def _calibrate(self, model, eval_steps: int = 100):
        dataset = VOCDataset(project.train_data_dir, is_test=False)

        predictor = create_mobilenetv2_ssd_lite_predictor(
            model, nms_method="hard", device=torch.device("cpu")
        )

        for i in tqdm(range(len(dataset)), total=eval_steps):
            if i == eval_steps:
                break
            image = dataset.get_image(i)
            boxes, labels, probs = predictor.predict(image)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i

    def uninplace(self, model):
        """Sets all `inplace` values to False"""
        if hasattr(model, "inplace"):
            model.inplace = False
        if not model.children():
            return
        for child in model.children():
            self.uninplace(child)

    def _prep_for_fusion(self, model, parent_name):
        """Fuses all conv+bn+relu, conv+bn, and conv+relu"""
        if not model.children():
            return []
        result = []
        candidate = []
        for name, child in model.named_children():
            new_name = parent_name + "." + name
            if new_name[0] == ".":
                new_name = new_name[1:]
            if type(child) == torch.nn.Sequential:
                candidate = []
                result.extend(self._prep_for_fusion(child, new_name))
            else:
                if len(candidate) == 0 and type(child) == torch.nn.Conv2d:
                    candidate = [new_name]
                elif len(candidate) == 1 and type(child) == torch.nn.ReLU:
                    candidate.append(new_name)
                    result.append(candidate)
                    candidate = []
                elif len(candidate) == 1 and type(child) == torch.nn.BatchNorm2d:
                    candidate.append(new_name)
                elif len(candidate) == 2:
                    if type(child) == torch.nn.ReLU:
                        candidate.append(new_name)
                    result.append(candidate)
                    candidate = []
        return result

    def scaled_quantization(self, model, scale=1e-3, zero_point=128):
        model.eval()
        self.uninplace(model)
        modules_to_fuse = self._prep_for_fusion(model, "")
        model.qconfig = torch.quantization.default_qconfig
        fused_model = torch.quantization.fuse_modules(
            model, modules_to_fuse, inplace=True
        )

        fused_model = torch.quantization.prepare(fused_model, inplace=True)
        torch.quantization.convert(fused_model, inplace=True)

    def save_model(self, model):
        file_name = helpers.get_time_of_day() + ".pth"
        torch.save(model.state_dict(), project.quantized_trained_model_dir / file_name)
        return file_name


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Evaluation Script")

    parser.add_argument(
        "--model-path", help="The model file to evaluate", required=True, type=str
    )
    args = parser.parse_args()
    model_path = args.model_path
    label_path = project.trained_model_dir / "voc-model-labels.txt"

    class_names = [name.strip() for name in open(label_path).readlines()]

    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    net.load(model_path)

    model_quantizer = ModelQuantizer()
    print("Quantizing the model")
    model_quantizer.quantize(net)
    model_quantizer.save_model(net)
    print("Saving the quantized model")
    file_name = model_quantizer.save_model(net)

    q_model = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    print("Quantinzed the vanilla model to get the correct definition")
    model_quantizer.quantize(q_model)
    print("Loading the quantized weights")
    q_model.load(project.quantized_trained_model_dir / file_name)

    dataset = VOCDataset(project.train_data_dir, is_test=False)

    predictor = create_mobilenetv2_ssd_lite_predictor(
        net, nms_method="hard", device=torch.device("cpu")
    )

    model_evaluator = model_evaluation.ModelEvaluator(
        description="Quantized model that is trained from Navya"
    )

    while len(tqdm._instances) > 0:
        tqdm._instances.pop().close()

    model_evaluator.evaluate(
        predictor=predictor,
        dataset=dataset,
        class_names=class_names,
        root_eval_dir=project.eval_results_dir,
        stopping_point=None,
    )
