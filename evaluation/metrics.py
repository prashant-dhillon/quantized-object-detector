import torch.nn as nn
import torch
from vision.utils import measurements
import numpy as np
import vision.utils.box_utils as box_utils
from thop.profile import profile
from Project import project
import os
import psutil
def calculate_average_sparsity(net):
    modules = net.modules()
    sum = 0
    considered = 0
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            if hasattr(layer, "weight") or hasattr(layer, "weight_orig"):
                considered += 1
                sum += (
                    100.0 * float(torch.sum(layer.weight == 0)) /
                    float(layer.weight.nelement())
                )
    if considered == 0:
        return 0
    return sum / considered


def compute_average_precision_per_class(
    num_true_cases, gt_boxes, difficult_cases, prediction_file, iou_threshold,
    use_2007_metric
):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()
        for i, image_id in enumerate(image_ids):
            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)
            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1

    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index
                              ] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(
                all_gt_boxes[class_index][image_id]
            )
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(
                all_gt_boxes[class_index][image_id]
            )
    return true_case_stat, all_gt_boxes, all_difficult_cases


def get_macs(net):
    dsize = ( 1 ,3, 300, 300)
    inputs = torch.randn(dsize)
    macs, params = profile(net, inputs)

    return macs / (1000**3)

def get_memory_used():
    pid=os.getpid()
    mem=psutil.Process(pid).memory_info()
    total=mem.rss/(1024.0 **2)
    return total

def get_size_of_model(model):
    """

    :param model:  nn.Module subclass
    :return: size of model in MB
    """
    torch.save(model.state_dict(), str(project.model_temp_dir / 'temp.p'))
    size = os.path.getsize(str(project.model_temp_dir / 'temp.p')) / 1e6
    os.remove(str(project.model_temp_dir / 'temp.p'))
    return size
