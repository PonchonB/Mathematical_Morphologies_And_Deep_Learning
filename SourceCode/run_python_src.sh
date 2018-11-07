#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_11_06_log_100_AsymAE_SparseNonNeg_MNIST >../Results/MNIST/AsymAE_infoGAN/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_11_06_dim100_firstMNIST &
deactivate

