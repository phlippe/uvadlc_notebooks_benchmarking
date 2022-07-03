#!/bin/bash

cd JAX/
for script in *.py
do
    python $script
done

cd ../PyTorch/
for script in *.py
do
    python $script
done