#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_12_11_log_100_AsymAE_MaxPlus_500epochs_WithoutDropout >../Results/AsymAE_MaxPlus/Simple/TestOutputs/18_12_11_dim100_500epochs_without_dropout &
deactivate

