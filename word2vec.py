"""
Word2Vec — Skip-Gram with Negative Sampling (SGNS)
Pure NumPy implementation - Dataset: WikiText-2

JetBrains Internship Task #1
"""

import re
import math
import pickle
import os
import time
from collections import Counter

import numpy as np

# ──────────────────────────────────────────────
# 1. Data Loading
# ──────────────────────────────────────────────

def load_wikitext2():
    """Download WikiText-2 via HuggingFace datasets and return raw text lines."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    lines = []
    for split in ("train", "validation", "test"):
        for row in ds[split]:
            text = row["text"].strip()
            if text:
                lines.append(text)
    return lines


def tokenize(lines):
    """Lowercase and split on non-alpha characters."""
    tokens = []
    for line in lines:
        tokens.extend(re.findall(r"[a-z]+", line.lower()))
    return tokens

# ──────────────────────────────────────────────
# 2. Vocabulary
# ──────────────────────────────────────────────

def build_vocab(tokens, min_count=5):
    """
    Build vocabulary from token list.

    Returns
    -------
    word2idx : dict  — word  -> integer id
    idx2word : list  — id    -> word
    freqs    : np.ndarray — normalised unigram frequencies indexed by id
    """
    counts = Counter(tokens)
    # keep only words with count >= min_count
    vocab = sorted([w for w, c in counts.items() if c >= min_count])
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = vocab
    total = sum(counts[w] for w in vocab)
    freqs = np.array([counts[w] / total for w in vocab], dtype=np.float64)
    return word2idx, idx2word, freqs

# ──────────────────────────────────────────────
# 3. Subsampling of Frequent Words
# ──────────────────────────────────────────────

def subsample(token_ids, freqs, t=1e-5):
    """
    Discard frequent tokens probabilistically.

    Each token with frequency f(w) is kept with probability:
        p_keep(w) = sqrt(t / f(w))
    """
    keep_prob = np.sqrt(t / freqs)          # one value per vocab word
    rng = np.random.default_rng()
    return [tid for tid in token_ids
            if rng.random() < keep_prob[tid]]

# ──────────────────────────────────────────────
# 4. Noise Distribution & Table
# ──────────────────────────────────────────────

def build_noise_table(freqs, table_size=10_000_000):
    """
    Pre-build a large integer table for O(1) negative sampling.

    Noise distribution: freq^0.75 (smoothed unigram).
    """
    powered = freqs ** 0.75
    powered /= powered.sum()
    table = np.zeros(table_size, dtype=np.int64)
    idx = 0
    cumulative = 0.0
    for word_id, p in enumerate(powered):
        cumulative += p
        upper = int(cumulative * table_size)
        upper = min(upper, table_size)       # safety clamp
        if upper > idx:
            table[idx:upper] = word_id
            idx = upper
    # fill any remaining slots (rounding gaps)
    if idx < table_size:
        table[idx:] = len(freqs) - 1
    return table


def sample_negatives(noise_table, k):
    """Draw k negative samples via the pre-built table — O(1) per sample."""
    indices = np.random.randint(0, len(noise_table), size=k)
    return noise_table[indices]

# ──────────────────────────────────────────────
# 5. Model (Embedding Matrices)
# ──────────────────────────────────────────────

def init_model(vocab_size, embed_dim):
    """
    Initialise two embedding matrices:
      W : (V, D) — center (input)  embeddings
      C : (V, D) — context (output) embeddings

    W is initialised uniformly in [-0.5/D, 0.5/D].
    C is initialised to zeros (standard practice).
    """
    W = (np.random.rand(vocab_size, embed_dim) - 0.5) / embed_dim
    C = np.zeros((vocab_size, embed_dim), dtype=np.float64)
    return W, C

# ──────────────────────────────────────────────
# 6. SGNS Step (Forward + Loss + Gradients + SGD)
# ──────────────────────────────────────────────

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def sgns_step(center_id, context_id, neg_ids, W, C, lr):
    """
    Perform one SGNS update for the pair (center, context)
    with K negative context samples.

    Loss
    ----
    L = log σ(c_pos · w) + Σ_k log σ(−c_neg_k · w)

    We MAXIMISE L via gradient ascent: θ ← θ + lr · ∂L/∂θ.

    Gradients
    ---------
    Using the identities:
        d/dx log σ(x)  = 1 − σ(x)
        d/dx log σ(−x) = −σ(x)

    Let s_pos = c_pos · w,  s_neg_k = c_neg_k · w.  Then:

        ∂L/∂s_pos   = 1 − σ(s_pos)
        ∂L/∂s_neg_k = −σ(s_neg_k)

    Chain rule (s = c · w):
        ∂L/∂w       = (1−σ(s_pos)) · c_pos + Σ_k (−σ(s_neg_k)) · c_neg_k
        ∂L/∂c_pos   = (1−σ(s_pos)) · w
        ∂L/∂c_neg_k = −σ(s_neg_k) · w

    Returns the loss (scalar) for logging.
    """
    K = len(neg_ids)

    # ---- forward ----
    w = W[center_id]                       # (D,)
    c_pos = C[context_id]                  # (D,)
    c_neg = C[neg_ids]                     # (K, D)

    s_pos = c_pos @ w                      # scalar
    s_neg = c_neg @ w                      # (K,)

    sig_pos = sigmoid(s_pos)               # scalar
    sig_neg = sigmoid(s_neg)               # (K,)

    # ---- loss ----
    # Add small eps for numerical safety in log
    eps = 1e-12
    loss = math.log(sig_pos + eps) + np.sum(np.log(1.0 - sig_neg + eps))

    # ---- gradients (ascent direction) ----
    # ∂L/∂w
    grad_w = (1.0 - sig_pos) * c_pos + ((-sig_neg)[:, None] * c_neg).sum(axis=0)

    # ∂L/∂c_pos
    grad_c_pos = (1.0 - sig_pos) * w

    # ∂L/∂c_neg_k
    grad_c_neg = (-sig_neg)[:, None] * w[None, :]   # (K, D)

    # ---- SGD update (gradient ASCENT — we maximise L) ----
    W[center_id]   += lr * grad_w
    C[context_id]  += lr * grad_c_pos
    C[neg_ids]     += lr * grad_c_neg

    return loss

# ──────────────────────────────────────────────
# 7. Training Loop
# ──────────────────────────────────────────────

def train(token_ids, W, C, noise_table, *,
          epochs=5, window=5, K=5,
          lr_start=0.025, lr_min=1e-4,
          log_every=100_000):
    """
    Train the skip-gram model with negative sampling.

    At each step:
      1. Pick a center token.
      2. Draw a random actual window size in [1, window].
      3. For each context position, run one sgns_step.

    The learning rate decays linearly from lr_start to lr_min
    over the total number of training steps.
    """
    n_tokens = len(token_ids)
    total_steps = epochs * n_tokens * window  # rough estimate
    global_step = 0
    running_loss = 0.0
    log_count = 0

    t0 = time.time()

    for epoch in range(1, epochs + 1):
        for i, center_id in enumerate(token_ids):
            # random window size
            actual_window = np.random.randint(1, window + 1)

            for delta in range(-actual_window, actual_window + 1):
                if delta == 0:
                    continue
                j = i + delta
                if j < 0 or j >= n_tokens:
                    continue

                context_id = token_ids[j]

                # draw negatives
                neg_ids = sample_negatives(noise_table, K)

                # linear LR decay
                progress = global_step / max(total_steps, 1)
                lr = lr_start - (lr_start - lr_min) * progress
                lr = max(lr, lr_min)

                loss = sgns_step(center_id, context_id, neg_ids, W, C, lr)
                running_loss += loss
                global_step += 1
                log_count += 1

                if global_step % log_every == 0:
                    elapsed = time.time() - t0
                    avg_loss = running_loss / log_count
                    wps = global_step / elapsed
                    print(f"  epoch {epoch}/{epochs}  "
                          f"step {global_step:>10,}  "
                          f"lr {lr:.6f}  "
                          f"loss {avg_loss:.4f}  "
                          f"words/s {wps:,.0f}")
                    running_loss = 0.0
                    log_count = 0

        elapsed = time.time() - t0
        print(f"Epoch {epoch} finished — {elapsed:.1f}s elapsed, "
              f"{global_step:,} total steps")

    return W, C

# ──────────────────────────────────────────────
# 8. Evaluation Utilities
# ──────────────────────────────────────────────

def normalize_rows(M):
    """L2-normalise each row of matrix M."""
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid division by zero
    return M / norms


def most_similar(word, W, word2idx, idx2word, topn=10):
    """Return the topn most similar words by cosine similarity.

    Embeddings are mean-centred before normalisation — this removes
    the dominant shared direction and produces much more
    discriminative cosine similarities.
    """
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return []
    W_centered = W - W.mean(axis=0)                # mean-centre
    W_norm = normalize_rows(W_centered)
    vec = W_norm[word2idx[word]]
    sims = W_norm @ vec                            # (V,)
    # exclude the query word itself
    sims[word2idx[word]] = -1.0
    top_ids = np.argsort(sims)[::-1][:topn]
    results = [(idx2word[i], sims[i]) for i in top_ids]
    return results


def analogy(a, b, c, W, word2idx, idx2word, topn=5):
    """
    Solve: a is to b as c is to ?
    vec = b - a + c  →  find nearest neighbour
    """
    for w in (a, b, c):
        if w not in word2idx:
            print(f"'{w}' not in vocabulary")
            return []
    W_centered = W - W.mean(axis=0)
    W_norm = normalize_rows(W_centered)
    vec = W_norm[word2idx[b]] - W_norm[word2idx[a]] + W_norm[word2idx[c]]
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    sims = W_norm @ vec
    for w in (a, b, c):
        sims[word2idx[w]] = -1.0
    top_ids = np.argsort(sims)[::-1][:topn]
    return [(idx2word[i], sims[i]) for i in top_ids]

# ──────────────────────────────────────────────
# 9. Main
# ──────────────────────────────────────────────

def main():
    # ── hyper-parameters ──
    EMBED_DIM   = 100
    WINDOW      = 5
    K           = 10      # negative samples
    EPOCHS      = 5
    MIN_COUNT   = 5
    LR_START    = 0.025
    LR_MIN      = 1e-4
    SUBSAMPLE_T = 1e-4    # less aggressive for a small dataset

    print("=" * 60)
    print("Word2Vec — Skip-Gram with Negative Sampling (pure NumPy)")
    print("=" * 60)

    # 1. load data
    print("\n[1/6] Loading WikiText-2 ...")
    lines = load_wikitext2()
    tokens = tokenize(lines)
    print(f"      Raw tokens: {len(tokens):,}")

    # 2. build vocab
    print("[2/6] Building vocabulary (min_count={}) ...".format(MIN_COUNT))
    word2idx, idx2word, freqs = build_vocab(tokens, min_count=MIN_COUNT)
    V = len(idx2word)
    print(f"      Vocabulary size: {V:,}")

    # 3. convert tokens to ids and subsample
    print("[3/6] Converting to IDs & subsampling ...")
    token_ids = [word2idx[t] for t in tokens if t in word2idx]
    print(f"      Tokens after vocab filter: {len(token_ids):,}")
    token_ids = subsample(token_ids, freqs, t=SUBSAMPLE_T)
    print(f"      Tokens after subsampling:  {len(token_ids):,}")

    # 4. noise table
    print("[4/6] Building noise table ...")
    noise_table = build_noise_table(freqs)

    # 5. init model
    print(f"[5/6] Initialising model — V={V}, D={EMBED_DIM}")
    W, C = init_model(V, EMBED_DIM)

    # 6. train
    print(f"[6/6] Training ({EPOCHS} epochs, window={WINDOW}, K={K}) ...\n")
    W, C = train(
        token_ids, W, C, noise_table,
        epochs=EPOCHS,
        window=WINDOW,
        K=K,
        lr_start=LR_START,
        lr_min=LR_MIN,
        log_every=100_000,
    )

    # ── save embeddings ──
    os.makedirs("embeddings", exist_ok=True)
    save_path = os.path.join("embeddings", "word2vec.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"W": W, "C": C, "word2idx": word2idx, "idx2word": idx2word}, f)
    print(f"\nEmbeddings saved to {save_path}")

    # ── use W + C for evaluation (Levy et al., 2015) ──
    E = W + C

    # ── quick evaluation ──
    print("\n" + "=" * 60)
    print("Evaluation — Most Similar Words  (embeddings = W + C, mean-centred)")
    print("=" * 60)
    for query in ("king", "computer", "water", "city", "good"):
        results = most_similar(query, E, word2idx, idx2word, topn=8)
        if results:
            neighbours = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {query:>10} → {neighbours}")

    print("\nEvaluation — Analogies (a:b :: c:?)")
    print("-" * 60)
    analogies = [
        ("king", "queen", "man"),
        ("paris", "france", "berlin"),
    ]
    for a, b, c in analogies:
        results = analogy(a, b, c, E, word2idx, idx2word, topn=5)
        if results:
            answers = ", ".join(f"{w} ({s:.3f})" for w, s in results)
            print(f"  {a}:{b} :: {c}:? → {answers}")

    print("\nDone.")


if __name__ == "__main__":
    main()
