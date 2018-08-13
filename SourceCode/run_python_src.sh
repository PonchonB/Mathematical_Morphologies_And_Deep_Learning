#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_12_log_100_noAMD_Simple_500epochs >../ShallowAE/Simple/TestOutputs/18_08_11_dim100_noAMD_500epochs &
deactivate
