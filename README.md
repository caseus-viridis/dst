# Dynamic sparse training

Training models with dynamic sparse parameters/activations in pytorch

## Setup

Do the following to set up.  

### The `dst` pack and dependencies

In your working environment:
```bash
python setup.py install
```

> **NOTE**: For developers of the `dst` pack do in your working environment:
> ```bash
> pip install -e .
> ```

### Environment variable

Point environment variable `$DATAPATH` to where your data directory or put in a `dst/.env` file:
```bash
DATAPATH=/where/datasets/are
```

## Overview of the `dst` pack

In the following we list the low-level contents of the pack for developers.  For high-level usage skip to the next section.  

```
src/dst
├── models
│   ├── char_rnn.py
│   ├── cifar_resnet.py
│   ├── cifar_wrn.py
│   ├── i1k_resnet.py
│   └── mnist_mlp.py
├── activation_sparse.py
├── modules.py
├── reparameterization.py
├── structured_dense.py
├── structured_sparse.py
└── utils.py
```

### Activation sparsity 
- All mechanisms that induce/handle activation sparsity are in `dst/activation_sparse.py`.

### Weight sparsity 
From lowest level up:
- All dense reparameterization mechanisms are in `dst/structured_dense.py`.  
- All sparse reparameterization mechanisms are in `dst/structured_sparse.py`.  The core low-level API is through a `StructuredSparseParameter` class that wraps a `StructuredDenseParameter` with a grouping mechanism.  
- Basic modules with dynamic sparse parameters are implemented in `dst/modules.py`, such as `DSLinear` and `DSConv2d`.  
- Model implementations are under `dst/models`.  
- The core high-level API is via a `DSModel` class in `dst/reparameterization.py`.  See next section for usage with an example.  


## Examples

### Training with dynamic sparse activations

A comparison of [Spatial bottleneck (SB) ResNet (Peng et al. 2018)](http://arxiv.org/abs/1809.02601) (i.e. structured spatial sparsity) against static/dynamic non-structured spatial sparsity.
```bash
python experiments/train_cifar_resnet.py -ds (cifar10|cifar100) -d (20|32|44|56|110) -sb (structured|static|dynamic) -q (1|2|3)
```

### Training with dynamic sparse weights

See `experiments/train_cifar_wrn.py` for a simple example as described in paper [Parameter efficient training of deep convolutional neural networks by dynamic sparse reparameterization (Mostafa & Wang 2018a)](https://openreview.net/pdf?id=S1xBioR5KX).

The following run trains a wide ResNet `WRN-28-2` on CIFAR10:
```bash
python experiments/train_cifar_wrn.py -ds cifar10 -w2
```
> **NOTE**: GPU required.

## General usage of the `dst` pack

### Activation sparsity

`dst.activation_sparse.SparseActivation` provides a base class for imposed activation sparsity through static or dynamic binary masking (i.e. input a dense activation tensor and output a sparse one), together with functionalities such as inspecting sparsity for book-keeping and Lp-norm computation that can be used for sparsity-inducing regularization.  

For example of handling sparse activations see `dst.activation_sparse.SparseBatchNorm`, which wraps a `SparseActivation` appending a batch-normalization operation after sparsification that only normalize non-zero elements.  

### Weight sparsity

Look into `experiments/train_cifar_wrn.py` for basic usage of dynamic sparse reparameterization as described in [Mostafa & Wang 2018a](https://openreview.net/pdf?id=S1xBioR5KX) and [2018b](https://openreview.net/pdf?id=BygIWTMdjX).  

At a highest level, one wraps a dynamic sparse model in a `DSModel` object like:
```python
from dst.reparameterization import DSModel

model = DSModel(
    model=my_dynamic_sparse_model,
    target_sparsity=0.9
)
```
Then one trains `model` as usual, during which one calls `model.reparameterize()` to do dynamic sparse reparameterization (i.e. reallocation of dense parameters).  

> **NOTE**: Hyperparameters of dynamic sparse reparameterization as described in the paper (slightly different) are:
> ```python
> target_fraction_to_prune=1e-2
> pruning_threshold=1e-3
> ```

According to the paper, dynamic sparse reparameterization `model.reparameterize()` is an atomic procedure consisting of three steps:
1. prune sparse weights by a global threshold, 
1. adjust pruning threshold, and
1. reallocate free non-zero parameters within and across layers

To experiment with these steps separately call `model.prune_by_threshold()`, `model.adjust_pruning_threshold()` and `model.reallocate_free_parameters()`, respectively.  

`DSModel` provides the following to allow inspection of sparseness statistics during training:
- `model.sum_table` gives model-level statistics, e.g. for the current example:
```
+--------------------+---------------------+--------------------+--------------------------------+----------+
| # total parameters | # sparse parameters | # dense parameters | # nonzero parameters in sparse | sparsity |
+--------------------+---------------------+--------------------+--------------------------------+----------+
|            1467610 |             1451520 |              16090 |                         130681 |   0.9000 |
+--------------------+---------------------+--------------------+--------------------------------+----------+
```
- `model.stats_table` gives breakdown statistics for each sparse parameter in the model, e.g. for the current example:
```
+---------------------------------+---------+-----------+----------+
|                Parameter tensor | # total | # nonzero | sparsity |
+---------------------------------+---------+-----------+----------+
| model.body.0.0.head.conv.weight |    4608 |      1655 |   0.6408 |
| model.body.0.0.tail.conv.weight |    9216 |      2401 |   0.7395 |
| model.body.0.1.head.conv.weight |    9216 |      1148 |   0.8754 |
| model.body.0.1.tail.conv.weight |    9216 |      1374 |   0.8509 |
| model.body.0.2.head.conv.weight |    9216 |      1771 |   0.8078 |
| model.body.0.2.tail.conv.weight |    9216 |      1853 |   0.7989 |
| model.body.0.3.head.conv.weight |    9216 |      1375 |   0.8508 |
| model.body.0.3.tail.conv.weight |    9216 |      2031 |   0.7796 |
| model.body.1.0.head.conv.weight |   18432 |      6379 |   0.6539 |
| model.body.1.0.tail.conv.weight |   36864 |      9904 |   0.7313 |
| model.body.1.1.head.conv.weight |   36864 |      7484 |   0.7970 |
| model.body.1.1.tail.conv.weight |   36864 |      7963 |   0.7840 |
| model.body.1.2.head.conv.weight |   36864 |      6574 |   0.8217 |
| model.body.1.2.tail.conv.weight |   36864 |      6460 |   0.8248 |
| model.body.1.3.head.conv.weight |   36864 |      7301 |   0.8019 |
| model.body.1.3.tail.conv.weight |   36864 |      5564 |   0.8491 |
| model.body.2.0.head.conv.weight |   73728 |      8358 |   0.8866 |
| model.body.2.0.tail.conv.weight |  147456 |     21292 |   0.8556 |
| model.body.2.1.head.conv.weight |  147456 |     11780 |   0.9201 |
| model.body.2.1.tail.conv.weight |  147456 |      8191 |   0.9445 |
| model.body.2.2.head.conv.weight |  147456 |      4008 |   0.9728 |
| model.body.2.2.tail.conv.weight |  147456 |      2489 |   0.9831 |
| model.body.2.3.head.conv.weight |  147456 |      1312 |   0.9911 |
| model.body.2.3.tail.conv.weight |  147456 |      2014 |   0.9863 |
+---------------------------------+---------+-----------+----------+
```

Just like `torch.nn.Module.parameters()` and `torch.nn.Module.named_parameters()`, `DSModel` provides `DSModel.sparse_parameters()` and `DSModel.named_sparse_parameters()` to iterate over all sparse parameter tensors for custom inspections and manipulations.  
