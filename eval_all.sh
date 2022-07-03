#!/bin/bash

cd JAX/
python Tutorial5_Inception_ResNet_DenseNet.py 2> ../logs/tutorial5_jax.txt
python Tutorial6_Transformers_and_MHAttention.py 2> ../logs/tutorial6_jax.txt

cd ../
cd PyTorch/
python Tutorial5_Inception_ResNet_DenseNet.py 2> ../logs/tutorial5_pytorch.txt
python Tutorial6_Transformers_and_MHAttention.py 2> ../logs/tutorial6_pytorch.txt
