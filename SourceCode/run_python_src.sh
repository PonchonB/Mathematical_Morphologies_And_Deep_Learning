#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_22_log_100_AMD_and_Originals_KLdivSum_500epochs >../ShallowAE/SeveralChannels/WithAMD/Sparse/KL_div_sum/TestOutputs/18_08_22_dim100_AMD_and_Originals_500epochs &
deactivate
