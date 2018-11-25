#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_11_25_log_100_ShallowAE_MaxPlus_Between0and1Constraint >../Results/ShallowAE_MaxPlus/NonNegativity/Between0and1Constraint/TestOutputs/18_11_25_dim100_firstMaxPlus &
deactivate

