---
title: "Multi-Vector Indexing in a Production RAG Platform"
description: "One idea from our Baltic DB&IS 2026 paper on a production RAG agent platform: storing each chunk under its own vector and under the questions it answers."
date: 2026-10-27 00:00:00 +0200
categories: [Research, RAG]
tags: [rag, retrieval, embeddings, document-expansion, llm]
image:
  path: /assets/img/posts/multi-vector/cover.png
  alt: "A user asks 'Can I end my subscription early?'. The question matches the generated question 'Can I cancel the contract early?', which resolves to the source chunk about terminating with 30 days' notice."
---

In June, Guntis Bārzdiņš and I published [*Engineering a Production RAG-Based Agent Platform*](https://ceur-ws.org/Vol-4211/paper05.pdf) at the Baltic DB&IS 2026 Conference Forum and Doctoral Consortium. The paper describes the architecture of Graip.AI Agent Labs, the production agent platform I work on at Graip.AI: agent configuration, document ingestion, query routing, hybrid retrieval, human-in-the-loop approval and evaluation.

This post covers one of the paper's two key ideas: **multi-vector indexing**, where every chunk of a document is stored under several vectors instead of one.

## The problem: users don't write like documents

Take a clause from a service agreement:

> Either party may terminate this agreement with 30 days' written notice.

A user of the knowledge base asks:

> Can I end my subscription early?

The clause answers the question, but the two texts share almost no words. One is a statement in legal language; the other is a short, informal question. Keyword search has nothing to match on. A dense embedding copes better with synonyms, but the chunk's single vector also has to represent everything else in the chunk, and a question and a statement land in different parts of the embedding space even when they are about the same thing.

This is an old problem. In 1987, Furnas, Landauer, Gomez and Dumais studied how people name the same objects and actions in software systems. Two people chose the same term less than 20% of the time. Their suggested fix was *unlimited aliasing*: let many different words lead to the same target [1].

Multi-vector indexing applies the same idea to retrieval. Instead of hoping the user phrases the question the way the document does, each chunk gets several entry points, written the way users ask.

## The idea: one chunk, several vectors

Multi-vector embedding is the last of five ingestion stages in the platform: parse, chunk, deduplicate, enrich, embed. For each chunk, the system stores:

1. **A direct embedding** of the chunk text. By this point the text is already enriched: an LLM prepends 50–100 tokens of document-level context, following Anthropic's *contextual retrieval* approach [2], so the chunk is not embedded in isolation.
2. **Three to five question embeddings.** An LLM writes questions the chunk can answer, and each question is embedded on its own.

That gives four to six vectors per chunk, all pointing to the same chunk. Dense search runs against all of them, and whichever vector matches, the result resolves to the source chunk. The generated questions are only used to find the chunk; the answer is generated from the chunk text.

![The user question matches a generated question for the chunk, while the single chunk vector is a weak match.](/assets/img/posts/multi-vector/figure-1.png){: .light width="800" height="531" }
![The user question matches a generated question for the chunk, while the single chunk vector is a weak match.](/assets/img/posts/multi-vector/figure-1-dark.png){: .dark width="800" height="531" }
_Figure 1. With one vector per chunk, the question has to match a legal clause. With multi-vector indexing, it matches a question generated for that clause._

Dense search is only half of the retrieval step. BM25 runs in parallel over the chunk text and catches exact terms such as product codes and acronyms, which embeddings handle poorly. The combined candidates are then graded for relevance before generation. Multi-vector indexing improves the dense half; it does not replace the lexical half.

## Why one vector is not enough

The idea has a research backstory. In an earlier shared task on reconstructing dictionary definitions from word embeddings, we found that a single fixed-size vector does not hold enough information to recover the full meaning of a word [3].

A chunk of a business document has the same problem at a larger scale. It may answer a question about notice periods, another about cancellation and a third about written form, and a single vector is a compromise between them. Separate question vectors let each of those "senses" of the chunk be found on its own.

## Where the idea comes from

Generating questions for documents at indexing time is not new. What changed is who writes the questions and how they are stored.

- **Document expansion with generated queries (doc2query).** A sequence-to-sequence model trained on query–passage pairs predicts queries for each document, and the predictions are appended to the document text before lexical indexing. The BEIR benchmark included its T5-based version, docT5query, among the retrievers it evaluated in zero-shot settings across domains [4].
- **Doc2Query--** showed the weak spot: generated queries can be hallucinated, which hurts effectiveness and inflates the index. Filtering the expansions with a relevance model improved effectiveness by up to 16% while shrinking the index [5].
- **HyDE** is the mirror image. It generates a hypothetical answer document for the query at search time and embeds that instead of the raw query [6]. Multi-vector indexing moves the generation to the document side and to indexing time, so it adds no LLM call at query time.
- **ColBERT** is multi-vector too, but at a different granularity: one vector per token, combined with late interaction [7]. Here, each vector is an ordinary embedding of one view of the chunk.

Compared with doc2query, the main change is the storage: the questions are not appended to one document but embedded as separate dense vectors, each pointing to the same chunk.

## What it costs

The vector index grows four to six times, LLM calls at ingestion double (context and questions for every chunk), and any change to chunking or embedding parameters means re-processing the whole knowledge base. The paper reports deployment experience, not a controlled ablation, so there is no single recall number to quote.

If you've tried question embeddings or document expansion in your own RAG system, I'd like to hear what worked and what didn't.

## Citation

```bibtex
@inproceedings{mukans-agents-2026,
  author    = {Mukans, Eduards and Barzdins, Guntis},
  title     = {Engineering a Production {RAG}-Based Agent Platform},
  booktitle = {Joint Proceedings of Baltic DB\&IS Conference Forum and
               Doctoral Consortium 2026},
  editor    = {Nikiforova, Anastasija and Iqbal, Mubashar and
               Kalibatien{\.e}, Diana and Kornyshova, Elena and
               Robal, Tarmo and Matulevi{\v{c}}ius, Raimundas and
               Liiv, Innar},
  series    = {CEUR Workshop Proceedings},
  volume    = {4211},
  pages     = {43--54},
  address   = {Tartu, Estonia},
  publisher = {CEUR-WS.org},
  year      = {2026},
  url       = {https://ceur-ws.org/Vol-4211/paper05.pdf},
}
```

## References

1. G. W. Furnas, T. K. Landauer, L. M. Gomez, S. T. Dumais. The vocabulary problem in human-system communication. *Communications of the ACM*, 30(11):964–971, 1987. [doi:10.1145/32206.32212](https://doi.org/10.1145/32206.32212)
2. Anthropic. Introducing contextual retrieval. Engineering blog, 2024. [anthropic.com](https://www.anthropic.com/engineering/contextual-retrieval)
3. E. Mukans, G. Strazds, G. Barzdins. RIGA at SemEval-2022 Task 1: Scaling recurrent neural networks for CODWOE dictionary modeling. In *Proceedings of SemEval-2022*, pp. 82–87. ACL, 2022. [ACL Anthology](https://aclanthology.org/2022.semeval-1.9/)
4. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In *NeurIPS 2021 Datasets and Benchmarks Track*, 2021. [PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/65b9eea6e1cc6bb9f0cd2a47751a186f-Paper-round2.pdf)
5. M. Gospodinov, S. MacAvaney, C. Macdonald. Doc2Query--: When less is more. In *Advances in Information Retrieval (ECIR 2023)*, pp. 414–422. Springer, 2023. [doi:10.1007/978-3-031-28238-6_31](https://doi.org/10.1007/978-3-031-28238-6_31)
6. L. Gao, X. Ma, J. Lin, J. Callan. Precise zero-shot dense retrieval without relevance labels. In *Proceedings of ACL 2023*. ACL, 2023. [doi:10.18653/v1/2023.acl-long.99](https://doi.org/10.18653/v1/2023.acl-long.99)
7. O. Khattab, M. Zaharia. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In *Proceedings of SIGIR 2020*, pp. 39–48. ACM, 2020. [doi:10.1145/3397271.3401075](https://doi.org/10.1145/3397271.3401075)
