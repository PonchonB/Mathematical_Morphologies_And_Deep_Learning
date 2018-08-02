#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_02_log_100_with_and_without_NonNegConstraint >../ShallowAE/Simple/18_08_02_dim100_with_and_without_NonNegConstraint &
deactivate
