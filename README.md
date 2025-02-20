# Intro

This project is the code of paper *Exploring Efficient Partial Differential Equation Solution Using Speed Galerkin Transformer*.

```latex
@inproceedings{10.1109/SC41406.2024.00084,
author = {Xun Wang, Zeyang Zhu, Siyu Zhang, Xianxi Zhu, Xiangyu Meng and Tao Song},
title = {Exploring Efficient Partial Differential Equation Solution Using Speed Galerkin Transformer},
year = {2024},
isbn = {9798350352917},
publisher = {IEEE Press},
url = {https://doi.org/10.1109/SC41406.2024.00084},
doi = {10.1109/SC41406.2024.00084},
booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis},
articleno = {78},
numpages = {14},
keywords = {Darcy Flow Equation, Fourier Neural Operator, Galerkin Attention, Model Acceleration Using GPUs},
location = {Atlanta, GA, USA},
series = {SC '24}
}
```

# Code Organization

```
SpGT
├── common
│   ├── path.py                    # Project path configuration, all components obtain path configuration from this file
│   └── trivial.py                 # Provide various trivial functions for implementation
├── storage
│   ├── data                       # Symbolic link pointing to the dataset storage directory
│   ├── model                      # Symbolic link pointing to the trained model storage directory
│   ├── evaluation                 # Directory for storing evaluation results
│   └── visualization              # Visualization images of experimental results
├── config
│   ├── config_accessor.py         # Access to configuration files
│   └── darcy_config.yaml          # Configuration file for Darcy problem
├── dataset
│   ├── data_accessor.py           # Access to data files
│   ├── darcy_dataset.py           # Dataset classes for the Darcy problem
│   └── darcy_generate             # Matlab code for generating the data required for the Darcy problem
├── network
│   ├── layer.py                   # Implementation of various layers in neural network models
│   ├── model.py                   # The constructed neural network module
│   ├── sp_layer.py                # Implementation of various layers of neural network models, optimized and accelerated versions
│   └── sp_model.py                # The constructed neural network module, optimized and accelerated versions
├── extension
│   ├── native                     # Optimization at the CUDA/C++level
│   └── bind                       # Encapsulate CUDA/C++ and provide PyTorch level API
├── engine
│   ├── metric.py                  # Loss function and metrices
│   ├── train.py                   # The training process iterates on a given number of epochs
│   └── darcy_engine.py            # The training or inference process of one epoch for the Darcy problem
├── run
│   ├── darcy_train.py             # Training of Darcy Model
│   ├── darcy_inference.py         # Inference of Darcy Model
│   ├── ddp_darcy_train_strong.py  # Training of Darcy model, using the DDP framework
│   └── ddp_darcy_inference.py     # Inference of Darcy Model, using the DDP framework
├── evaluate                       # Various evaluations for optimizations
└── visualize                      # Visualization of evaluation results
```

# Prerequisites and Datasets

The primary software environment includes: Python 3.9, CUDA 11.8 (including cuBLAS and cuDNN libraries), and PyTorch 2.1.0.

Generally speaking, the optimization work in this paper is not heavily dependent on specific software versions, but differences in implementation efficiency across versions may affect optimization outcomes.

Fetch the latest release version from this repository and place the code under a directory. Ensure that the top-level directory of the project is named SpGT and the path to the project is in the Python interpreter's search path. 

To ensure the correct path, we use ROOT_PATH in the SpGT/common/path.py script to specify the path of the project's top-level directory (i.e., the path of SpGT). Users can modify this path.

To avoid excessive repository size, we store the dataset files and trained models outside the project and use symbolic links to point to the actual resource directories. Before running the scripts, users need to create symbolic links named SpGT/storage/data and SpGT/storage/model, pointing to the directories where the dataset and trained models are actually stored, respectively.

Use the SpGT/dataset/darcy_generate/gen.m file to generate the necessary dataset. Alternatively, you can download the Darcy equation data with a resolution of $421\times421$ from the link https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing. If you choose to use the downloaded $421\times421$ resolution dataset, you can skip this step.

To generate the required Darcy equation dataset, run the SpGT/dataset/darcy_generate/gen.m script using MATLAB. The generated dataset will be stored in the same directory. You can use num_file to specify the number of files to generate, num_data to specify the number of data points in each file, and S to specify the resolution of each data point.

It is important to note that in the provided generation script, the coefficients and solutions of the equation are labeled with 'a' and 'u', whereas the dataset provided by the FNO authors uses 'coeff' and 'sol' to label the coefficients and solutions. The difference is only in naming, so be mindful of this when reading the data. We have specified the correct key values in the corresponding code location, so use the appropriate keys as needed.

# How To Use

## Train and Inference

Use the SpGT/run/darcy_train.py script to train the complete model, and use the SpGT/run/darcy_inference.py script to perform inference with the trained model.

The script SpGT/run/darcy_train.py is used to train the model. After the training is completed, the model will be stored in the SpGT/storage/model directory. The training script first reads all configuration parameters, which are specified in the SpGT/config/darcy_config.yaml file. Users can modify these configurations. Generally, the configurations that need to be specified are as follows:

- num_data: The number of sample points in the dataset.
- fine_resolution: The resolution of each sample point in the dataset.
- subsample_node: The sampling scale for the entire model, representing that every subsample_node×subsample_node points from the fine_resolution are sampled as one data point.
- subsample_attn: The sampling scale for the Galerkin Attention part of the model, with the same function as subsample_node.
- train_dataset and valid_dataset: The file names of the training dataset and validation dataset, respectively.
- name_module: The name of the model used; specify GT to use the unoptimized model and SpGT to use the optimized model.

The script SpGT/run/darcy_inference.py uses the trained model for inference, with the model's checkpoint file name specified. This allows verification of the model's solution accuracy. Additionally, the subsample_node and subsample_attn parameters in the configuration file can be modified to perform inference at higher resolutions, verifying the Fourier operator's ability to learn super-resolution.

In the SpGT/run/ddp_sh directory, bash scripts are provided for submitting distributed training. These scripts require cluster support with the Slurm distributed resource management and job scheduling system. They use Slurm commands to submit distributed jobs, for example, sbatch --gpus=4 ddp_strong_gpu4.sh.

## Evaluation

Use various scripts in the SpGT/evaluate directory to perform comprehensive performance evaluations.

In the SpGT/evaluate directory, several scripts are provided for comprehensive experimental evaluation of the proposed optimization methods. You can directly run the SpGT/evaluate/sp_all.py script to execute all experimental evaluations. The resolution_list specifies the resolutions to be tested, and the batch_list specifies the batch sizes to be tested. The specified ranges are designed to fully utilize the 32GB memory of a V100 GPU. If you encounter out-of-memory errors during execution, please reduce the test range.

The experimental results will be written to the SpGT/storage/evaluation directory. Additionally, we provide a SpGT/storage/evaluation/time.tgz file, which contains the results obtained from our experiments. These results are also the data sources used in our paper.

## Visualization

Use the scripts in the SpGT/visualize/plot directory to visualize the experimental evaluation results.

In the SpGT/visualize/plot directory, several plotting scripts are provided. These scripts use the matplotlib library to visualize the experimental results and save the images to the SpGT/storage/visualization directory.
