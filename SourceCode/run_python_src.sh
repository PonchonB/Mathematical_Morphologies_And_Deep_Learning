#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_10_30_log_100_AsymAE_SparseNonNeg_Hoyer >../Results/AsymAE_infoGAN/Sparse_NonNeg/Hoyer_NonNegConstraint/TestOutputs/18_10_30_dim100_first_Hoyer &
deactivate

