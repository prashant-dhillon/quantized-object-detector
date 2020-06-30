import torch.nn as nn
import torch
import time
from models.predictor import Predictor
from hyper_parameters import mobilenetv1_ssd_config as config
from models.mobilenet_v2 import MobileNetV2
from models.SSDLite import GraphPath
from torch.nn import ModuleList
from models.InvertedResidual import InvertedResidual
# from models.SeperableConv2d import get_seperable_conv2d
from models.SeperableConv2d import SeparableConv2d
from torch.nn import Conv2d
from models.SSDLite import SSD


def create_mobilenetv2_ssd_lite_predictor(
    net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device("cpu")
):
    predictor = Predictor(
        net,
        config.image_size,
        config.image_mean,
        config.image_std,
        nms_method=nms_method,
        iou_threshold=config.iou_threshold,
        candidate_size=candidate_size,
        sigma=sigma,
        device=device,
    )
    return predictor


def create_mobile_net_v2(width_mult=1.0, use_batch_norm=True):
    base_net = MobileNetV2(width_mult=width_mult, use_batch_norm=use_batch_norm)
    return base_net


def create_mobilenetv2_ssd_lite(
    num_classes,
    width_mult=1.0,
    use_batch_norm=True,
    is_test=False,
    device=None,
):
    base_net = MobileNetV2(
        width_mult=width_mult, use_batch_norm=use_batch_norm
    ).features

    source_layer_indexes = [GraphPath(14, "conv", 3), 19]
    extras = ModuleList(
        [
            InvertedResidual(1280, 512, stride=2, expand_ratio=0.2),
            InvertedResidual(512, 256, stride=2, expand_ratio=0.25),
            InvertedResidual(256, 256, stride=2, expand_ratio=0.5),
            InvertedResidual(256, 64, stride=2, expand_ratio=0.25),
        ]
    )

    regression_headers = ModuleList(
        [
            SeparableConv2d(
                in_channels=round(160 * width_mult),
                out_channels=6 * 4,
                kernel_size=3,
                padding=1,
            ),
            SeparableConv2d(
                in_channels=1280, out_channels=6 * 4, kernel_size=3, padding=1
            ),
            SeparableConv2d(
                in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1
            ),
            SeparableConv2d(
                in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1
            ),
            SeparableConv2d(
                in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1
            ),
            Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=1),
        ]
    )

    classification_headers = ModuleList(
        [
            SeparableConv2d(
                in_channels=round(160 * width_mult),
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1,
            ),
            SeparableConv2d(
                in_channels=1280,
                out_channels=6 * num_classes,
                kernel_size=3,
                padding=1
            ),
            SeparableConv2d(
                in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            SeparableConv2d(
                in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            SeparableConv2d(
                in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1
            ),
            Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=1),
        ]
    )

    return SSD(
        num_classes,
        base_net,
        source_layer_indexes,
        extras,
        classification_headers,
        regression_headers,
        is_test=is_test,
        config=config,
        device=device,
    )


def str2bool(s):
    return s.lower() in ("true", "1")


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        del self.clock[key]
        return interval


def save_checkpoint(
    epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path
):
    torch.save(
        {
            "epoch": epoch,
            "model": net_state_dict,
            "optimizer": optimizer_state_dict,
            "best_score": best_score,
        },
        checkpoint_path,
    )
    torch.save(net_state_dict, model_path)


def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
