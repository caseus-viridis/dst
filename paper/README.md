# Dynamic parameter reallocation in deep CNNs

The code implements the experiments in the ICLR 2019 submission: [***Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization***](https://openreview.net/forum?id=S1xBioR5KX)

## Introduction

A deep convnet (CNN) can be pruned to a fraction of its original size with little to no degradation in accuracy. A new network having the same size as the pruned model, however, can not be trained *de novo* to match the accuracy of the compressed model. It thus seems that starting with a big network followed by pruning is essential to getting a small network with good performance. Previously, however, only small networks with static parameterizations were considered. Question is, can a small network with dynamic re-parameterization train to match the performance of an equivalent-size network that was obtained by pruning a big model? The answer is yes. We introduce a dynamic re-parameterization scheme that re-allocates parameters within and across sparse layers. In deep CNNs, this scheme outperforms static parameterization schemes and has a very small computational overhead.

## Instructions

This code implements different variations of dynamic re-parameterization including DeepR as well as static parameterizations based on tied parameters similar to [the HashedNet paper](https://arxiv.org/abs/1504.04788). It also implements iterative pruning where it can take a dense model and prune it down to the required sparsity. 

The main python executable is `variable_sized_nws.py`. Results are saved under a `./runs/` directory created at the invocation directory. An invocation of `variable_sized_nws.py` will save various accuracy metrics as well as the model parameters in the file `./runs/{model name}_{job idx}`. Accuracy figures as well as several diagnostics are also printed out. 

### General usage
```shell
variable_sized_nws.py [-h] [--epochs EPOCHS] [--model {mnist_mlp,cifar10_WideResNet,imagenet_resnet50}] [-b BATCH_SIZE] [--momentum MOMENTUM] [--nesterov NESTEROV] [--weight-decay WEIGHT_DECAY]
                             [--L1-loss-coeff L1_LOSS_COEFF] [--print-freq PRINT_FREQ] [--layers LAYERS] [--start-pruning-after-epoch START_PRUNING_AFTER_EPOCH]
                             [--prune-epoch-frequency PRUNE_EPOCH_FREQUENCY] [--prune-target-sparsity-fc PRUNE_TARGET_SPARSITY_FC] [--prune-target-sparsity-conv PRUNE_TARGET_SPARSITY_CONV]
                             [--prune-iterations PRUNE_ITERATIONS] [--post-prune-epochs POST_PRUNE_EPOCHS] [--n-realloc-params N_REALLOC_PARAMS] [--threshold-prune] [--prune] [--validate-set]
                             [--grow-across-layers] [--tied] [--rewire] [--no-validate-train] [--DeepR] [--DeepR_eta DEEPR_ETA] [--stop-rewire-epoch STOP_REWIRE_EPOCH] [--rewire-fraction REWIRE_FRACTION]
                             [--sub-kernel-granularity] [--sparse-resnet-downsample] [--conv-group-lasso] [--big-new-weights] [--widen-factor WIDEN_FACTOR] [--initial-sparsity-conv INITIAL_SPARSITY_CONV]
                             [--initial-sparsity-fc INITIAL_SPARSITY_FC] [--job-idx JOB_IDX] [--data DIR] [-j N] [--resume RESUME] [--schedule-file SCHEDULE_FILE] [--name NAME]
```
Optional arguments:
```
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --model {mnist_mlp,cifar10_WideResNet,imagenet_resnet50}
                        network name (default: mnist_mlp)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        mini-batch size (default: 128)
  --momentum MOMENTUM   momentum
  --nesterov NESTEROV   nesterov momentum
  --weight-decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        L2 weight decay coefficient (default: 1e-4)
  --L1-loss-coeff L1_LOSS_COEFF
                        Lasso coefficient (default: 0.0)
  --print-freq PRINT_FREQ, -p PRINT_FREQ
                        print frequency (default: 10)
  --layers LAYERS       total number of layers for wide resnet (default: 28)
  --start-pruning-after-epoch START_PRUNING_AFTER_EPOCH
                        Epoch after which to start pruning (default: 20)
  --prune-epoch-frequency PRUNE_EPOCH_FREQUENCY
                        Interval (in epochs) between prunes (default: 2)
  --prune-target-sparsity-fc PRUNE_TARGET_SPARSITY_FC
                        Target sparsity when pruning fully connected layers (default: 0.98)
  --prune-target-sparsity-conv PRUNE_TARGET_SPARSITY_CONV
                        Target sparsity when pruning conv layers (default: 0.5)
  --prune-iterations PRUNE_ITERATIONS
                        Number of prunes. Set to 1 for single prune, larger than 1 for gradual pruning (default: 1)
  --post-prune-epochs POST_PRUNE_EPOCHS
                        Epochs to train after pruning is done (default: 10)
  --n-realloc-params N_REALLOC_PARAMS
                        Target number of parameters to reallocate each prune/grow cycle (default: 600)
  --threshold-prune     Prune based on a global adaptive threshold and not a fixed fraction from each layer (default: False)
  --prune               prune mode to sparsify a dense model (default: False)
  --validate-set        whether to use a validation set or not (default: False)
  --grow-across-layers  Move weights between layers in the prune/grow cycle. (default: False)
  --tied                whether to use tied weights instead of sparse ones, i.e, similar to hash nets (default: False)
  --rewire              whether to run parameter re-allocation (default: False)
  --no-validate-train   whether to run validation on training set (default: False)
  --DeepR               DeepR mode (default: False)
  --DeepR_eta DEEPR_ETA
                        eta coefficient for DeepR (default: 0.1)
  --stop-rewire-epoch STOP_REWIRE_EPOCH
                        Epoch after which to stop rewiring (default: 1000)
  --rewire-fraction REWIRE_FRACTION
                        Fraction of weight to rewire. Only effective if threshold-prune is false (default: 0.1)
  --sub-kernel-granularity
                        Use sub-kernel granularity while rewiring(default: False)
  --sparse-resnet-downsample
                        Use a sparse/tied tensor for the resnet downsampling convolution(default: False)
  --conv-group-lasso    Use group lasso to penalize an entire kernel patch(default: False)
  --big-new-weights     Use weights initialized from the initial distribution for the new connections instead of zeros(default: False)
  --widen-factor WIDEN_FACTOR
                        widen factor for wide resnet (default: 10)
  --initial-sparsity-conv INITIAL_SPARSITY_CONV
                        Initial sparsity of conv layers(default: 0.5)
  --initial-sparsity-fc INITIAL_SPARSITY_FC
                        Initial sparsity for fully connected layers(default: 0.98)
  --job-idx JOB_IDX     job index provided by the job manager
  --data DIR            path to imagenet dataset
  -j N, --workers N     number of data loading workers (default: 8)
  --resume RESUME       path to latest checkpoint (default: none)
  --schedule-file SCHEDULE_FILE
                        yaml file containing learning rate schedule and rewire period schedule
  --name NAME           name of experiment
```

### Specific experiments

The three yaml files : `mnist_experiments.yaml`, `wrnet_experiments.yaml`, and `resnet_experiments.yaml` contain YAML lists of all the invocations of the python executable needed to run all the experiments in the paper's main text. 

Extra experiment files `wrnet_experiments_deepr.yaml` and `wrnet_experiments_SET.yaml` contain the invocation commands for some additional experiments we carried out in response to an anonymous commenter on [the OpenReview forum](https://openreview.net/forum?id=S1xBioR5KX). They use the DeepR algorithm by [Bellec at al. 2018](https://arxiv.org/abs/1711.05136) and the SET algorithm by [Mocanu et  al. 2018](https://www.nature.com/articles/s41467-018-04316-3).

### Important notes

- Code development and all experiments were done with Python 3.6 and pytorch 0.4.1. 
- All experiments were conducted on NVidia TitanXP GPUs.
- Imagenet experiments require multi-GPU data parallelism, which is done by default using all available GPUs specified by environment variable `CUDA_VISIBLE_DEVICES`.