# L0-ARM

This repository contains the code for [L0-ARM: Network Sparsification via Stochastic Binary Optimization](https://arxiv.org/abs/1904.04432).

## Demo
Visualization of part of the neurons in conv-layer(left) and fully-connected layer(right) of the LeNet-5-Caffe sparsified by L0-ARM. To achieve computational efficiency, only neuron-level (instead of weight-level) sparsification is considered. 

<p align="center">
    <img height="400" alt="conv_layer" src="https://github.com/leo-yangli/l0-arm/blob/master/conv_layer.gif?raw=true"/>
    <img height="400" alt="fc_layer" src="https://github.com/leo-yangli/l0-arm/blob/master/fc_layer.gif?raw=true"/>
</p>

## Requirements
    pytorch>1.0.0
    tnt
    fire
    tqdm
    numpy
    tensorboardX

## Usage
    python main.py <function> [--args=value]
        <function> := train | test | help
    example: 
        python main.py train --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --optimizer=adam --lr=0.001
        python main.py test --model=ARMLeNet5 --dataset=mnist --lambas="[.1,.1,.1,.1]" --load_file="checkpoints/ARMLeNet5_2019-06-19 14:27:03/0.model"
        python main.py train --model=ARMWideResNet --dataset=cifar10 --lambas=.001 --optimizer=momentum --lr=0.1 --schedule_milestone="[60, 120]"
        python main.py help
        
## Citation
If you found this code useful, please cite our paper.

    @inproceedings{l0arm2019,
      title={{L0-ARM}: Network Sparsification via Stochastic Binary Optimization},
      author={Yang Li and Shihao Ji},
      booktitle={The European Conference on Machine Learning (ECML)},
      year={2019}
    }
