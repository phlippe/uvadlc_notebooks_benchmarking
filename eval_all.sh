#!/bin/bash

mkdir -p logs

cd JAX/
python Tutorial5_Inception_ResNet_DenseNet.py
python Tutorial6_Transformers_and_MHAttention.py

cd ../
cd PyTorch/
python Tutorial5_Inception_ResNet_DenseNet.py
python Tutorial6_Transformers_and_MHAttention.py
