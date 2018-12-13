#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_12_13_log_100_AsymAE_infoGAN_NonNeg_500epochs >../Results/AsymAE_infoGAN/NonNegativity/NonNegConstraint/TestOutputs/18_12_13_dim100_500epochs &
deactivate

