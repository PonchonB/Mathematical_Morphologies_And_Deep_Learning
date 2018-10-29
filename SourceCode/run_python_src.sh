#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_10_25_log_100_AsymAE_SparseNonNeg_withAMD_noOriginals >../Results/AsymAE_infoGAN/SeveralChannels/WithAMD_NoOriginals/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_10_25_dim100_AMD_NoOriginals &
deactivate

