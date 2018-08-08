#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_07_log_100_AMD_Simple_noOriginal_and_normalized >../ShallowAE/SeveralChannels/WithAMD_NoOriginals/Simple/TestOutputs/18_08_07_dim100_AMD_noOriginals_Normalized_channels &
deactivate
