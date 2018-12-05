#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_12_05_log_100_ShallowAE_MaxPlus_Sparse_NonNeg_500epochs_WithDropout_50_PlusInsteadOfMinus >../Results/ShallowAE_MaxPlus/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_12_05_dim100_500epochs_with_dropout_50 &
deactivate

