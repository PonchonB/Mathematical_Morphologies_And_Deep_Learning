#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_30_log_100_AsymAE_infoGAN_simple >../AsymAE_infoGAN/Simple/TestOutputs/18_08_30_dim100_noAMD_firstTest &
deactivate
