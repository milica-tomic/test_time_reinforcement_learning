# Test Time Reinforcement Learning

# Learning to Reason with Small Models

Implementation of the Word2Vec training loop from scratch using only NumPy with Adam optimizator. 

## Project Overview

This project implements the Skip-Gram with Negative Sampling (SGNS) variant of Word2Vec, as described in Mikolov et al. (2013b). The implementation covers the optimization procedure: forward pass, loss computation, gradient derivation, and parameter updates (with NumPy).

## Requirements

- Python 3.8+
- `numpy`
- `os`
- `json`
- `zipfile`
- `urllib`


## Instructions

First, if you don't have some of the libraries mentioned in the [Requirements](#requirements) section, please do as follows:
```
pip install numpy
```
After you check that all libraries are installed, run the script directly:

```
python word2vec.py
```

The [Text8](https://mattmahoney.net/dc/text8.zip) and [Google Analogy](https://github.com/tmikolov/word2vec) datasets will download automatically during execution. 

## Model Architecture

The model consists of two embedding matrices (both of shape `(V, D)`):

- **`W_in`** — center-word embeddings, used as final word vectors  
- **`W_out`** — context-word embeddings, used only during training

Both matrices are initialized with small random uniform values:

```python
W_in  = (random(V, D) - 0.5) / D
W_out = (random(V, D) - 0.5) / D
```

Only `W_in` is used for evaluation. The difference, compared to Mikolov et al. (2013b), is the initialization of `W_out`. In this implementation, it is initialized with random values, whereas in the original paper, it is initialized only with zeros. This choice was made to facilitate faster symmetry breaking and prevent numerical plateaus during the initial stages of training, similar to the approach used in **Gensim**.

## Training Procedure

### Skip-Gram Objective

For each center word $v_c$, predict its surrounding context words within a window.  
The **dynamic window** $c \sim \text{Uniform}[1, \text{window}]$ gives closer words a higher effective weight.

Instead of pre-calculating every single word pair, a generator is implemented that yields batches as needed. This makes the training much more RAM-friendly, even with large corpora.

### Negative Sampling Loss (SGNS)

Instead of a full softmax over the vocabulary, SGNS replaces the problem with $K$ binary classification tasks:

$$\mathcal{L} = -\left[ \log \sigma(\mathbf{u}_o \cdot \mathbf{v}_c) + \sum_{k=1}^{K} \log \sigma(-\mathbf{u}_{n_k} \cdot \mathbf{v}_c) \right]$$

where $\sigma$ is the sigmoid function, **u_o** is the true context vector, **u_n_k** are $K$ noise word vectors sampled from the $\text{unigram}^{0.75}$ distribution.

This reduces the per-step cost from $O(V \cdot D)$ (full softmax) to $O(K \cdot D)$.

### Gradient Derivation

Applying the chain rule through the sigmoid $\sigma(x)$, using $\frac{d}{dx}\log\sigma(x) = 1 - \sigma(x)$:

**Center word vector** $\mathbf{v}_c$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = \underbrace{(\sigma(s_{pos}) - 1)}_{\text{positive error}} \cdot \mathbf{u}_o + \sum_{k=1}^{K} \underbrace{\sigma(s_{neg_k})}_{\text{negative error}} \cdot \mathbf{u}_{n_k}$$

**True context vector** $\mathbf{u}_o$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = (\sigma(s_{pos}) - 1) \cdot \mathbf{v}_c$$

**Noise vector** $\mathbf{u}_{n_k}$:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{n_k}} = \sigma(s_{neg_k}) \cdot \mathbf{v}_c$$

The error signals $(\sigma - 1)$ and $\sigma$ drive the updates: when the model is correct, the errors approach zero and the update vanishes.

These gradients map directly to the code:

```python
err_pos = (sig_pos - 1.0) 
err_neg = sig_neg 

grad_vc    = err_pos[:, None] * u_pos + np.einsum('bk,bkd->bd', err_neg, u_neg)
grad_u_pos = err_pos[:, None] * vc
grad_u_neg = np.einsum('bk,bd->bkd', err_neg, vc)
```

### Parameter Updates

Two optimizers are supported, selectable via the `optimizer` argument in `train()`:

**SGD** with linear learning rate decay (as in the original paper):

```python
lr = max(lr_start * (1 - processed / total), lr_min)
```

Updates use `np.add.at` instead of plain indexing to correctly accumulate gradients when the same word index appears multiple times in a batch:

```python
np.add.at(W_in,  centers,  -lr * grad_vc)
np.add.at(W_out, contexts, -lr * grad_u_pos)
```

**Adam** (Sparse / Lazy Adam) — updates only the rows of the moment matrices that were touched in the current batch, keeping the update efficient for sparse embedding lookups. Because Adam adapts the learning rate per parameter, it requires a much lower starting learning rate than SGD:

```python
OPTIMIZER = "adam"
LR_START  = 0.005  # Adam requires lower LR than in the original paper
```

### Subsampling of Frequent Words

Frequent words (the, of, a) are discarded with probability:

$$P(\text{keep}) = \sqrt{\frac{t}{f(w)}}, \quad t = 10^{-5}$$

This reduces corpus size and improves vector quality for rare words.

### Noise Distribution

Negative samples are drawn from the $\text{unigram}^{0.75}$ distribution:

$$P(w) \propto f(w)^{0.75}$$

The exponent 0.75 reduces the dominance of frequent words as negatives without fully equalising to a uniform distribution.


## Batch Vectorisation

The training loop processes `B` pairs simultaneously instead of one by one (in the original Word2Vec C implementation processes samples one by one). This shifts the computational load from the slow Python level to NumPy's highly optimized C backend. 

The mathematics are identical to the per-sample case — dimensions are larger (due to vectorisation).

The negative dot products require `np.einsum`:

```python
s_neg = np.einsum('bd,bkd->bk', vc, u_neg)
# for each b: s_neg[b,k] = vc[b] · u_neg[b,k]
```

## Evaluation

### 1. Nearest Neighbours

Cosine similarity between a query vector and all vectors in `W_in` is used.
The model was trained on approximately 4M tokens (after subsampling
from the full 17M token Text8 corpus).

| Query | Top neighbours |
|---|---|
| `king` | throne, vii, kings, prince, crowned |
| `paris` | bologna, villa, seine, cimeti, venice |
| `computer` | computing, computers, hardware, graphics, machines |
| `music` | musical, jazz, musicians, techno, folk |

Despite training on a relatively small corpus, the model captures
meaningful semantic clusters.

Note that `vii` appear due to frequent co-occurrence with
  royal names (Henry VIII, Louis VII) in Wikipedia text — a direct
  consequence of distributional semantics rather than explicit encoding.

### 2. Analogy Accuracy

Tests the linear structure of the embedding space: **a − b + c = ?**

```
king - man + woman  →  queen    ✓
paris - france + germany  →  berlin   ✓
```

Accuracy = % of analogies where top-1 prediction matches the expected word.

The benchmark used is the original Google word analogy dataset (Mikolov et al., 2013a), containing 19,544 analogies split into semantic categories (e.g. capital cities, currency) and syntactic categories (e.g. plural forms, verb tenses). Words absent from the vocabulary are skipped and counted separately.

**Result: 2.1% accuracy (198/9310 evaluated, 10234 skipped) with Adam optimizer**

This low accuracy is expected and consistent with the literature.
The original Word2Vec paper reports ~70% accuracy, but was trained on
100 billion tokens, approximately 25,000x more data than Text8 (~4M tokens).

Possible directions for improving results include using Top-5 metrics instead of Top-1, increasing the number of epochs, and experimenting with the Adam optimizer for potentially faster convergence.

## Additional 

In the following repositories, [learning_to_reason_with_small_models](https://github.com/milica-tomic/learning_to_reason_with_small_models) and [hallucionation_detection](https://github.com/milica-tomic/hallucination_detection) (here, you can see the results using SGD for parameter updates), you can see some other modifications to the skip-gram with negative sampling implementation.

## References

- Mikolov, T. et al. (2013a). *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781
- Mikolov, T. et al. (2013b). *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013
