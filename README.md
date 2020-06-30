# Quantized-object-detector
To start with this repo, it's very recommended to setup a linux environment with some kind of virtual envrionments 

Initializing the env

1 - Install python 3.7
 
2 - `pip install -r requirements.txt`

3 - Download VOC 2007 & VOC 2012 & put them on the `dataset/` directory with the following structure

- dataset/  
    - VOC/
        -  2007
            -  train/
            -  test/  
        - 2012
    
    
        

## Folder Structure 
- Dataset VOC 2007 - VOC 2012 should go into the `dataset/` directory 
- Trained models are following this structure 
-trained_models/ 
    - mb2-ssd-lite-mp-0_686.pth   # Original Google pretrained model
    - voc-model-labels.txt # VOC Dataset labels
    - /pruned
        - All the pruned models will be saved here
    - /quantized
        - All the quantized models will be saved here
    - /new-def-trained
        - Holds the trained model with the new definition from navya team
- evaluation/ Holds the evaluation scripts
- quantization/ holds the quantization scripts 
- pruning/ holds the pruning scripts 
- models/ holds the definitions and some utilities for creating the models
- eval_results/ outout of the evaluation scripts will be saved here
- training/ holds the training scripts 
- Project.py is used to reference any directory in the project


## Running Evaluation
Evaluation scripts takes only 1 argument which is the model to evaluate

To use this script you can run the following command
```bash
python evaluation/model_evaluation.py --model-path ./trained_models/new-def-trained/mb2-ssd-lite-Epoch-119-Loss-4.837355399743105.pth
```

## Running Pruning
Note: After pruning is done, the evaluation directly happens.

Pruning script takes 2 arguments

1- model path to prune

2- Amount of pruning ( float between 0 & 1 )

To use this script you can run the following command
```bash
 python pruning/pruner.py --model-path ./trained_models/new-def-trained/mb2-ssd-lite-Epoch-119-Loss-4.837355399743105.pth --amount 0.2
```

## Running Quantization
The quantization script takes 1 argument which is the model path to quantize 

To use this script you can run the following command
```bash
python quantization/quantization.py --model-path ./trained_models/new-def-trained/mb2-ssd-lite-Epoch-119-Loss-4.837355399743105.pth 
```


# References
## PyTorch Implementations of SSD-Object Detector & References
- [[Original Repo](https://github.com/qfgaohao/pytorch-ssd)]
- Tutorial covering SSD : [[blog](https://lilianweng.github.io/lil-log/2018/12/27/object-detection-part-4.html)]

## Quantization References
- PyTorch Quantization desin proposal [[link](https://github.com/pytorch/pytorch/wiki/torch_quantization_design_proposal)]
- Tests within PyTorch to validate Quantization [[git](https://github.com/pytorch/pytorch/blob/master/test/test_quantization.py)]
- x86 optimization for quantized models [[git](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized), [CPU](https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu)]
- Quantization Tutorial in PyTorch [[link](https://pytorch.org/tutorials/#quantization-experimental)]
- PyTorch seems to have a few tutorials on quantization already up on [[Colab](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)]


## References Quantized-Object detection
- Data-Free Quantization Through Weight Equalization and Bias Correction [[pdf](https://arxiv.org/pdf/1906.04721v3.pdf)], [[code]](https://github.com/ANSHUMAN87/Bias-Correction)]
- Fully Quantized Network for Object Detection CVPR 2019 [[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Fully_Quantized_Network_for_Object_Detection_CVPR_2019_paper.pdf)]
- Distilling Object Detectors with Fine-grained Feature Imitation [[pdf](https://arxiv.org/pdf/1906.03609.pdf)]
