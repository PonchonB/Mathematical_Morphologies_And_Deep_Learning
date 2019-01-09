#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_12_18_log_100_AsymAE_infoGAN_Max_Plus_500epochs_Dropout50 >../Results/AsymAE_MaxPlus/Dropout50/Simple/TestOutputs/18_12_18_dim100_500epochs &
deactivate

