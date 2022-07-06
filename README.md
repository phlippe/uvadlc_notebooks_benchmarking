# Benchmarking UvA Deep Learning Tutorials
Benchmark scripts for comparing the [UvA Deep Learning Tutorials](https://uvadlc-notebooks.readthedocs.io/en/latest/) in PyTorch and JAX. The scripts are reduced versions of the tutorials, and save the runtimes of each model in a logging file in the `logs/` directory.

## Preparations

The scripts have been executed with PyTorch v1.11 and JAX v0.3.13. To fully reproduce the environment, you can use the provided conda environment file `environment.yml`. Additionally, for JAX, CUDA 11.7 and cuDNN 8.4 have to be installed. Other CUDA version above 11.3 are expected to work similarly.

## Running benchmarks

To run all benchmarks, simply run `bash eval_all.sh`. Afterwards, you can find the evaluated runtimes in the `logs/` folder.

## Benchmark times

We report the runtimes for all models on an NVIDIA RTX3090 GPU with 24-core CPU. The times are averaged over two runs, which commonly have a standard deviation of <5%.

### Tutorial 5: Inception, ResNet and DenseNet

| Models                |   PyTorch   |     JAX     |
|-----------------------|:-----------:|:-----------:|
| GoogleNet             | 53min 50sec | 16min 10sec |
| ResNet                | 20min 47sec |  7min 51sec |
| Pre-Activation ResNet | 20min 57sec |  8min 25sec |
| DenseNet              | 49min 23sec |  20min 1sec |

### Tutorial 6: Transformers and Multi-Head Attention

| Models            |   PyTorch   |     JAX    |
|-------------------|:-----------:|:----------:|
| Reverse Sequence  | 0min 26sec  | 0min 7sec  |
| Anomaly Detection | 16min 34sec | 3min 45sec |

### Tutorial 9: Deep Autoencoders

| Models           |   PyTorch   |     JAX    |
|------------------|:-----------:|:----------:|
| AE - 64 latents  | 13min 10sec | 7min 10sec |
| AE - 128 latents | 13min 11sec | 7min 10sec |
| AE - 256 latents | 13min 11sec | 7min 11sec |
| AE - 384 latents | 13min 12sec | 7min 14sec |

### Tutorial 11: Normalizing Flows

| Models                  |      PyTorch     |        JAX       |
|-------------------------|:----------------:|:----------------:|
| MNIST Flow - Simple     | 2hrs 37min 29sec | 1hrs 17min 59sec |
| MNIST Flow - VarDeq     | 3hrs 25min 10sec | 1hrs 36min 56sec |
| MNIST Flow - Multiscale | 2hrs 17min 10sec |      57min 57sec |

### Tutorial 15: Vision Transformer

| Models                |   PyTorch   |     JAX     |
|-----------------------|:-----------:|:-----------:|
| Vision Transformer    | 28min 40sec | 27min 10sec |
