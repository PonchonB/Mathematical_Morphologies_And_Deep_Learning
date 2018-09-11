#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
source /home/ponchon/StageENV/bin/activate
nohup python main.py 2>../Logs/18_09_10_log_100_ShallowAE_SparseNonNeg_withPADO_withOriginals >../Results/ShallowAE/SeveralChannels/WithPADO/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_09_10_dim100_PADO_originals &
deactivate

