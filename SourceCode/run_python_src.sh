#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_01_log_testKLsum >../ShallowAE/Sparse/KL_div_sum/TestOutputs/18_08_01_testKLsum &
deactivate
