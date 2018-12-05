#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_12_03_log_100_ShallowAE_MaxPlus_500epochs_WithoutDropout_PlusInsteadOfMinus_Sparse_NonNegConstraint >../Results/ShallowAE_MaxPlus/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_12_03_dim100_500epochs_without_dropout &
deactivate

