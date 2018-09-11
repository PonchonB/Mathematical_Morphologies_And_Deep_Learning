#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_09_11_log_100_ShallowAE_Simple_withPADO_withOriginals >../Results/ShallowAE/SeveralChannels/WithPADO/Simple/TestOutputs/18_09_11_dim100_PADO_originals &
deactivate

