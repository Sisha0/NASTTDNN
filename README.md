# Neural architecture search for tabular deep neural networks

The goal of the project is to apply neural architecture search (NAS) to deep neural networks (DNNs) specified for tabular data. We take a multi-layer perceptron [[1]](#1) and apply it to California Housing [[2]](#2) and Covertype [[3]](#3) datasets. To select an optimal architecture, we use DARTS [[4]](#4).

# Project structure and setup

We have two jupyter notebooks: one for [classification](./mlp_classification.ipynb) task (Covertype) and one for [regression](./mlp_regression.ipynb) (California housing) task. Before running the code from notebooks, you should install the needed dependencies from the root of the project:

```bash
pip install -r requirements.txt
```

[Dataset](./dataset/) and [evaluators](./evaluators/) folders contain utilities to preprocess datasets and run NAS experiments. We use the same preprocessing techniques as in the original paper on tabular deep neural networks [[1]](#1). As a reference we've also used these tutorials:

1) [Revisiting deep learning models for tabular data](https://github.com/yandex-research/rtdl-revisiting-models/blob/main/package/example.ipynb)
2) [DARTS tutorial from nni library](https://github.com/microsoft/nni/blob/master/examples/tutorials/darts.py#L297)
3) [NAS tutorial from nni library](https://github.com/microsoft/nni/blob/master/examples/tutorials/hello_nas.py)

[Models](./models/) folder contains sources for MLP model and selected search space for NAS.
<!-- 1) Hidden dimension for linear layers from the `[32, 64, 128, 256, 512]` grid.
2) Number of blocks in the `[1, 10]` range (except one in_block, so `[2, 11]` blocks in total). Each is a sequence of linear, activation and dropout layers.
3) Activation type within each block. One of `nn.ReLU, nn.GELU, nn.SiLU`. -->

# References
<a id="1">[1]</a> 
Gorishniy, Y., Rubachev, I., Khrulkov, V., & Babenko, A. (2021). Revisiting deep learning models for tabular data. Advances in Neural Information Processing Systems, 34, 18932-18943.

<a id="2">[2]</a>
R. Kelley Pace and R. Barry. Sparse spatial autoregressions. Statistics & Probability Letters, 33(3):
291–297, 1997.

<a id="3">[3]</a>
J. A. Blackard and D. J. Dean. Comparative accuracies of artificial neural networks and discriminant
analysis in predicting forest cover types from cartographic variables. Computers and Electronics
in Agriculture, 24(3):131–151, 2000.

<a id="4">[4]</a>
Liu, H., Simonyan, K., & Yang, Y. (2018). Darts: Differentiable architecture search. arXiv preprint arXiv:1806.09055.
