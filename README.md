# Benchmarking UvA Deep Learning Tutorials
Benchmark scripts for comparing the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/) in PyTorch and JAX. The scripts are reduced versions of the tutorials, and save the runtimes of each model in a logging file in the `logs/` directory.

## Preparations

The scripts have been executed with PyTorch v1.11 and JAX v0.3.13. To fully reproduce the environment, you can use the provided conda environment file `environment.yml`. Additionally, for JAX, CUDA 11.7 and cuDNN 8.4 have to be installed. Other CUDA version above 11.3 are expected to work similarly.

## Running benchmarks

To run all benchmarks, simply run `bash eval_all.sh`. Afterwards, you can find the evaluated runtimes in the `logs/` folder.
