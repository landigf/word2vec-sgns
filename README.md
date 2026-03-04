# Word2Vec — Skip-Gram with Negative Sampling (SGNS)

Pure NumPy implementation · Dataset: WikiText-2 (via HuggingFace `datasets`)

> JetBrains Internship — Task #1

---

## Quick Start

```bash
pip install -r requirements.txt
python word2vec.py
```

The script will:
1. Download WikiText-2 automatically
2. Build vocabulary, subsample frequent tokens
3. Train for 5 epochs (takes ~15–30 min on a laptop CPU)
4. Save embeddings to `embeddings/word2vec.pkl`
5. Print nearest-neighbour and analogy evaluations

---

## Model Architecture

Two embedding matrices are learned:

| Matrix | Shape | Role |
|--------|-------|------|
| **W** | (V, D) | Center (input) embeddings |
| **C** | (V, D) | Context (output) embeddings |

After training, **W** is used as the final word embeddings.

### Why two separate matrices?

If we tied `W = C`, the model could trivially maximise the positive score `w · w = ‖w‖²` by inflating vector norms, without learning any meaningful word relationships. Separate matrices force the model to encode genuine semantic structure.

---

## Objective Function

For a center word *w* with true context word *c_pos* and *K* negative samples *c_neg_1, …, c_neg_K*:

$$
\mathcal{L} = \log \sigma(c_{\text{pos}} \cdot w) + \sum_{k=1}^{K} \log \sigma(-c_{\text{neg}_k} \cdot w)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

We **maximise** this objective (gradient *ascent*).

### Why not full softmax?

Full softmax computes $\exp(v_i \cdot w)$ for every word $i$ in the vocabulary — **O(V · D)** per training pair. With $V \approx 20\,000$ and $D = 100$, that is 2 million multiplications per step. Negative sampling reduces cost to **O(K · D)** — roughly 4,000× cheaper for $K = 5$.

---

## Gradient Derivation

Starting from the useful identities:

$$
\frac{d}{dx} \log \sigma(x) = 1 - \sigma(x)
$$

$$
\frac{d}{dx} \log \sigma(-x) = -\sigma(x)
$$

Define the scalar scores:

$$
s_{\text{pos}} = c_{\text{pos}} \cdot w, \qquad s_{\text{neg}_k} = c_{\text{neg}_k} \cdot w
$$

Then:

$$
\frac{\partial \mathcal{L}}{\partial s_{\text{pos}}} = 1 - \sigma(s_{\text{pos}})
$$

$$
\frac{\partial \mathcal{L}}{\partial s_{\text{neg}_k}} = -\sigma(s_{\text{neg}_k})
$$

Applying the chain rule ($s = c \cdot w$):

| Parameter | Gradient (ascent direction) |
|-----------|---------------------------|
| $\dfrac{\partial \mathcal{L}}{\partial w}$ | $(1 - \sigma(s_{\text{pos}})) \, c_{\text{pos}} + \displaystyle\sum_k \bigl(-\sigma(s_{\text{neg}_k})\bigr) \, c_{\text{neg}_k}$ |
| $\dfrac{\partial \mathcal{L}}{\partial c_{\text{pos}}}$ | $(1 - \sigma(s_{\text{pos}})) \, w$ |
| $\dfrac{\partial \mathcal{L}}{\partial c_{\text{neg}_k}}$ | $-\sigma(s_{\text{neg}_k}) \, w$ |

Parameters are updated via SGD with gradient **ascent**:

$$
\theta \leftarrow \theta + \eta \, \frac{\partial \mathcal{L}}{\partial \theta}
$$

---

## Key Design Decisions

### Noise distribution: `freq^0.75`

Raw unigram frequency would flood negatives with trivially easy words like "the" and "a". Raising frequency to the power 0.75 smooths the distribution, boosting the probability of rarer words. This produces harder negatives and stronger gradient signal. The exponent 0.75 was found empirically by [Mikolov et al. (2013)](https://arxiv.org/abs/1310.4546).

### Subsampling of frequent words

Each token of word $w$ is **discarded** with probability:

$$
P_{\text{discard}}(w) = 1 - \sqrt{\frac{t}{f(w)}}
$$

where $f(w)$ is the word's corpus frequency and $t = 10^{-5}$. This dramatically accelerates training and improves embedding quality for content words by removing noise from high-frequency function words.

### Random window size

At each training step, a window size is drawn uniformly from $[1, W_{\max}]$. This effectively down-weights distant context positions (which are noisier), giving nearby context words more influence on average.

### O(1) noise table

`np.random.choice(V, p=noise_dist)` is O(V) per call. We pre-build a 10M-element integer table where each word appears proportionally to `freq^0.75`. Sampling then reduces to a single random integer index — O(1).

### Numerically stable sigmoid

```python
np.where(x >= 0, 1/(1+exp(-x)), exp(x)/(1+exp(x)))
```

For large positive `x`, `exp(-x)` underflows harmlessly to 0. For large negative `x`, `exp(x)` underflows harmlessly to 0. Neither branch overflows.

---

## Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBED_DIM` | 100 | Embedding dimensionality |
| `WINDOW` | 5 | Maximum context window size |
| `K` | 5 | Number of negative samples per positive pair |
| `EPOCHS` | 5 | Number of training epochs |
| `MIN_COUNT` | 5 | Minimum word frequency for vocabulary inclusion |
| `LR_START` | 0.025 | Initial learning rate |
| `LR_MIN` | 0.0001 | Minimum learning rate (floor for linear decay) |
| `SUBSAMPLE_T` | 1e-5 | Subsampling threshold |

---

## Possible Optimisations

- **Batch updates:** Process multiple (center, context) pairs as matrix operations — enables SIMD and potential GPU offloading.
- **Adagrad / Adam:** Adaptive learning rates give large updates to rare words (sparse gradient problem is severe in word2vec).
- **Hierarchical softmax:** Replace negative sampling with a Huffman tree — theoretically appealing but typically slower in practice.
- **Cython / Numba JIT:** The inner loop is Python-level; a compiled extension would yield 10–50× speedup.

---

## Project Structure

```
word2vec-sgns/
├── word2vec.py          # Full implementation: data, model, training, evaluation
├── requirements.txt     # numpy, datasets
├── README.md            # This file
├── .gitignore           # Excludes embeddings/, __pycache__, etc.
└── embeddings/          # Created at runtime — contains saved .pkl files
```

---

## References

- Mikolov, T., et al. *Efficient Estimation of Word Representations in Vector Space.* arXiv:1301.3781, 2013.
- Mikolov, T., et al. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS, 2013.
- Goldberg, Y. & Levy, O. *word2vec Explained.* arXiv:1402.3722, 2014.
