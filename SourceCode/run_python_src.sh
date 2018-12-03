#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_11_30_log_100_ShallowAE_MaxPlus_Sparse_NonNeg_500epochs_WithDropout >../Results/ShallowAE_MaxPlus/Sparse_NonNeg/KLdivSum_Between0and1Constraint/TestOutputs/18_11_30_dim100_500epochs_with_dropout &
deactivate

