# ADR 0004 — Hybrid retrieval fused by rank, not by score

**Status:** accepted · 2026-09-03

## Context

Vector search is good at paraphrase and bad at rare exact tokens: ask for `ERR_4021` and
the embedding returns a neighbourhood of similar-looking identifiers, none of them the one
requested. Lexical search has the mirror failure — it cannot match "how do I reset my
password" against "Credential recovery procedure".

Enterprise corpora are full of both kinds of query. Error codes, part numbers, ticket ids
and policy clause references sit alongside natural-language questions.

## Decision

Run both retrievers and fuse the rankings with **Reciprocal Rank Fusion**
(`score = Σ 1/(k + rank)`, `k = 60`), then diversify with **Maximal Marginal Relevance**.

## Why rank and not score

A BM25 score of 14.2 and a cosine similarity of 0.81 are not comparable quantities.
Normalising them requires per-corpus calibration that drifts as the corpus grows. Ranks are
always comparable, and `k = 60` flattens the contribution curve so that appearing in both
lists outweighs placing first in one — agreement between independent retrievers is stronger
evidence than confidence from one.

## Why MMR afterwards

The top five results of a good retriever are frequently five near-copies of the same
paragraph. Filling a context window with the same fact five times is how a system with
excellent retrieval metrics still produces an answer missing what the user needed.

## Consequences

- Both indexes must be maintained, and the lexical index is rebuilt rather than updated
  incrementally: BM25's IDF depends on corpus-wide document frequencies, so incremental
  updates leave scores subtly wrong in a way that never surfaces as an error.
- Retrieval over-fetches by 3× before fusion, because RRF and MMR both need more candidates
  than the caller asked for.
- **Both indexes must be tenant-partitioned.** The lexical index was not, initially. The
  vector store being correct hid it, because the leak only appeared in hybrid mode.
- Cost: more moving parts than a single vector search, and a fusion constant that is
  defensible but not tuned per corpus.