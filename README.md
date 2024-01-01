# Part of Speech Tagging
This repo contains tutorials covering how to perform part-of-speech (PoS) tagging using [PyTorch](https://github.com/pytorch/pytorch) 2.1 using Python 3.11 and using MATLAB. The most important part of the code is from scratch implementation of the backpropagation of the transformer.

## Getting Started
To install PyTorch, see installation instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).
To download, conll 2003 dataset formatted for this code (thanks to Hugging Face!): 
* [Training set](https://drive.google.com/file/d/1PTfU4nI6aKrV9xsASFbOUf6Lkwxo1eD9/view?usp=sharing)
* [Testing set](https://drive.google.com/file/d/1RS4QIIv6TpCfden6bONfC1I4YqsJsBjA/view?usp=sharing)
* [Validation set](https://drive.google.com/file/d/1pkBoTOc1VE9kqGeGsjq57AgOAqjU6f0M/view?usp=sharing)

To download, word vectors trained using word2vec:
* [word vectors](https://drive.google.com/file/d/1v4VAsPCz6vqXrDqcF91i0okUnxZN3W_H/view?usp=sharing)

## Results

Number of Parameters : 202351

Training Accuracy : 93.59%

Testing Accuracy : 89.62%

## References
Here are some things I looked at while making this.
* [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
* [Karpathy's backpropagation video](https://www.youtube.com/watch?v=q8SA3rM6ckI)
