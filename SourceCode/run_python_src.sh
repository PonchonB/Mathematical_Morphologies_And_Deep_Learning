#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_08_06_log_100_KLsum_NonNegConstraint_withAMD >../ShallowAE/SeveralChannels/WithAMD/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_08_06_dim100_KLdivSum_NonNegConstraint_withAMD &
deactivate
