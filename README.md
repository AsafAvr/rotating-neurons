# Disentangling MLP Neuron Weights in Vocabulary Space

Code for **ROTATE** (Rotation-Optimized Token Alignment in weighT spacE), from

> Asaf Avrahamy, Yoav Gur-Arieh, Mor Geva. *Disentangling MLP Neuron Weights in Vocabulary Space.*
> arXiv:2604.06005, 2026. [[paper]](https://arxiv.org/abs/2604.06005)

<a href="https://colab.research.google.com/github/AsafAvr/rotating-neurons/blob/main/notebooks/rotate_one_neuron.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

ROTATE decomposes an MLP neuron into **vocabulary channels**: directions in weight space
whose projection onto the vocabulary is sparse and interpretable. It is data-free and needs
no forward passes, only the neuron's weights and the model's (un)embedding matrix.

The key observation is that a weight vector encoding a single, coherent concept has **high
kurtosis** when projected onto the vocabulary.
ROTATE optimises rotations of a neuron's weight vector to maximise this kurtosis, which recovers one channel.
It then masks that channel's tokens and repeats to find the next. Experiments on Gemma-2-2B-it and
Llama-3.1-8B-Instruct show that the channels are faithful to the neuron's behaviour, and
that neuron descriptions built from them beat activation-based descriptions 2–3× in
head-to-head comparisons.

## Try it

[`notebooks/rotate_one_neuron.ipynb`](notebooks/rotate_one_neuron.ipynb) walks through the method
on one neuron, the paper's running example (Gemma-2-2B-it, layer 18, neuron 9005), or any neuron of
Gemma or Llama. It runs on a free Colab T4 GPU: only the neuron's weights and the unembedding matrix
are downloaded, never the full model.

## Install

```bash
pip install git+https://github.com/AsafAvr/rotating-neurons
# or, for development:
git clone https://github.com/AsafAvr/rotating-neurons.git && cd rotating-neurons && pip install -e ".[dev]"
```


## License

Code: MIT ([LICENSE](LICENSE)). Model weights and anything derived from them, including
channel vectors, are subject to the Gemma and Llama 3.1 licences.
