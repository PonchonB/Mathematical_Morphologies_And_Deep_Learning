#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
nohup python main.py 2>../Logs/18_07_12_log_testShallowAEwithAMD >../ShallowAE/WithAMD/Simple/TestOutputs/18_07_12_testShallowAEwithAMD &
