#!/bin/bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
source /home/ponchon/StageENV/bin/activate
nohup python main2.py 2>../Logs/18_09_11_log_100_svm_max_approx_AsymAE_NonNeg_SParse >../Results/AsymAE_infoGAN/Sparse_NonNeg/KLdivSum_NonNegConstraint/TestOutputs/18_09_11_dim100_svm_max_approx &
deactivate

