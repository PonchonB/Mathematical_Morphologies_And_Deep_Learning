#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source StageENV/bin/activate
nohup python main.py 2>../Logs/18_07_31_log_testKLsum >../ShallowAE/Sparse/KL_div_sum/TestOutputs/18_07_12_testShallowAEwithAMD &
deactivate