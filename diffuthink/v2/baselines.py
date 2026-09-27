"""Bidirectional bigram baseline using only visible immediate neighbours."""
import numpy as np
from .model import PAD, BOS, MASK


def bigram_accuracy(train, noisy, targets, selected, vocab_size):
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    unigram = np.ones(vocab_size, dtype=np.float64)
    for start in range(0, len(train), 2048):
        rows = np.asarray(train[start:start+2048], dtype=np.int64)
        left, right = rows[:, :-1].reshape(-1), rows[:, 1:].reshape(-1)
        valid = (left != PAD) & (right != PAD)
        np.add.at(counts, (left[valid], right[valid]), 1)
        unigram += np.bincount(rows[rows != PAD], minlength=vocab_size)
    unigram[[PAD, BOS, MASK]] = 1e-9
    prior = unigram / unigram.sum()
    row_total = counts.sum(-1) + 1
    correct = total = 0
    for source, truth, mask in zip(noisy, targets, selected):
        for i in np.flatnonzero(mask):
            # P(candidate | left) * P(right | candidate), with a unigram
            # backoff if a neighbour is masked. No clean neighbour is consulted.
            probabilities = prior.copy()
            if i > 0 and source[i-1] not in (PAD, MASK):
                probabilities = (counts[source[i-1]].astype(np.float64) + prior) / row_total[source[i-1]]
            if i+1 < len(source) and source[i+1] not in (PAD, MASK):
                probabilities *= (counts[:, source[i+1]].astype(np.float64) + prior[source[i+1]]) / row_total
            probabilities[[PAD, BOS, MASK]] = 0
            correct += int(probabilities.argmax() == truth[i])
            total += 1
    return {"policy": "bidirectional_bigram", "masked_accuracy": correct / total, "masked_tokens": total,
            "description": "Add-one unigram-prior smoothing; immediate visible neighbours only"}
