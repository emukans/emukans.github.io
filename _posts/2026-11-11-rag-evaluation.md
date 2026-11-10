---
title: "Evaluating a Production RAG Agent: RAG Metrics, LLM Judges and MLflow"
description: "The third idea from our Baltic DB&IS 2026 paper: how Graip.AI Agent Labs, a production RAG agent platform, is evaluated. Software tests show that the system works; RAG metrics and LLM judges, run offline and online in MLflow, show that its answers are worth something."
date: 2026-11-11 00:00:00 +0200
categories: [Research, RAG]
tags: [rag, evaluation, llm-as-judge, ragas, mlflow, testing, llm]
image:
  path: /assets/img/posts/rag-evaluation/cover.png
  alt: "A user asks 'Can I end the contract early?' and the agent answers 'Yes, with 30 days' notice', citing the contract. Two checks pass: a software test confirms the citation is present, and an LLM judge confirms the claim is supported."
---

The previous posts covered two ingestion ideas from our paper [*Engineering a Production RAG-Based Agent Platform*](https://ceur-ws.org/Vol-4211/paper05.pdf) [1]: [multi-vector indexing](/posts/multi-vector/) and [contextual enrichment](/posts/contextual-enrichment/). Both change what goes into the index of Graip.AI Agent Labs, the production agent platform I work on at Graip.AI. This post is about the question that comes after any such change: **does the system still work, and are its answers better and not worse?** Software tests answer the first half. ML evaluation answers the second, and it is what most of this post is about.

## Software tests: does the system work?

We have the usual unit, integration and end-to-end tests, and they check structure, not meaning: chunk token budgets, the number of candidates after rank fusion, valid JSON, citations present.

One detail is specific to systems built on external services. In the integration tests, responses from LLMs [1] and from other third-party services are recorded once in a baseline run and replayed from a local cache afterwards. This makes the tests deterministic and independent of those services, so they can run on every commit. End-to-end tests call the live services and run nightly and before a release.

A replayed test proves that the code handles a recorded response correctly. It says nothing about whether the answer is any good. That is the second half of the question.

## What to measure

In the platform, a question is routed, hybrid retrieval and re-ranking select the top chunks, and an LLM writes the answer from them, which is then verified against the sources [1]. The quality of an answer depends on two things: whether the right chunks reached the LLM, and whether the LLM used them correctly.

The standard metrics measure neither well. Retrieval metrics such as nDCG and MRR reward putting the right document first, because they model a person reading results from the top down. The LLM reads all chunks at once, so what matters is what is in the set: an outdated chunk at position 5 barely changes nDCG, but it can change the answer, because LLMs tend to trust false context [2]. Text-overlap metrics such as BLEU and ROUGE count shared words [4]: "The contract cannot be terminated with 30 days' notice" shares almost every word with "The contract can be terminated with 30 days' notice" and says the opposite.

So we measure retrieval and generation separately, with metrics that judge the content of the chunks and the meaning of the answer instead of rank and wording.

## Four metrics, one per pipeline stage

We use four metrics as defined in RAGAS [3]. Each one points at a different stage of the pipeline, so a drop in one score tells you where to look.

![The pipeline runs route, retrieve, re-rank, generate and verify. Context precision and context recall sit under retrieve and re-rank, faithfulness under generate and verify, and answer relevance spans the whole pipeline. Only context recall needs a reference answer.](/assets/img/posts/rag-evaluation/figure-1.png){: .light width="800" height="521" }
![The pipeline runs route, retrieve, re-rank, generate and verify. Context precision and context recall sit under retrieve and re-rank, faithfulness under generate and verify, and answer relevance spans the whole pipeline. Only context recall needs a reference answer.](/assets/img/posts/rag-evaluation/figure-1-dark.png){: .dark width="800" height="521" }

Faithfulness is the most useful for catching hallucinations: RAGAS splits the answer into short claims and asks an LLM whether each one is supported by the context [3]. It is also the best-validated, agreeing with human annotators in 95% of cases on the RAGAS authors' dataset, against 78% for answer relevance and 70% for context relevance [3]. We treat faithfulness and context recall above 0.8 as strong [1].

## The LLM as judge

All four metrics are computed by an LLM. Strong LLM judges correlate with human judgement far better than BLEU or ROUGE [4] and agree with human preferences above 80% of the time, about as often as humans agree with each other [5]. They also have known biases: they prefer the first of two answers, longer answers and answers written by their own model [5].

So people don't score answers; they prepare the data. Domain experts write the questions for the evaluation dataset and annotate the expected result for each one, and the judge compares against that. Shopify does the same with at least three experts per sample, and uses their agreement as the bar for the judge: iterating on the judge's prompt raised its Cohen's kappa with the experts from 0.02 to 0.61, against 0.69 between the experts themselves [7].

## Offline and online

The judges run in two modes, and the difference is the dataset. Offline, the judge compares each answer with the expected result, so a drop in a score can only come from what you changed; context recall can run only here. The dataset doesn't have to be big to start: Anthropic suggests 20 to 50 tasks drawn from real failures [6]. Online, there is no expected result, and the judge rates each production response on its own from 1 to 5, for example for completeness, faithfulness or whether the agent picked the right tool. It is a weaker signal, but it covers questions nobody put in the dataset.

![Offline, the judge compares the actual answer with an expected answer annotated by a person and finds a match. Online, there is no expected answer; the judge rates an answer about the hotel limit abroad 2 out of 5 for completeness because the limit is missing. A low online score goes to human review, gets an expected answer and is added to the offline dataset.](/assets/img/posts/rag-evaluation/figure-2.png){: .light width="800" height="604" }
![Offline, the judge compares the actual answer with an expected answer annotated by a person and finds a match. Online, there is no expected answer; the judge rates an answer about the hotel limit abroad 2 out of 5 for completeness because the limit is missing. A low online score goes to human review, gets an expected answer and is added to the offline dataset.](/assets/img/posts/rag-evaluation/figure-2-dark.png){: .dark width="800" height="604" }

How many traces are scored is configured per assistant and client. Safety and PII checks run on every trace and can block a response. Quality judges run separately on a sample of completed traces, so they add no latency; about 400 scored traces put the share of low scores within ±5 percentage points at 95% confidence, and at low traffic that means scoring everything. The lowest-scoring traces are reviewed periodically and become new offline records once an expert adds the expected result. User feedback comes separately, through a feedback form.

After our own tests we mostly use Gemini Flash and Flash-Lite as judges, for their ratio of quality to price. The generating model is configured per assistant, mostly from the GPT, Gemini and Claude families, and picked by its offline scores; when it is also a Gemini model, the judge's preference for its own family [5] is worth keeping in mind.

## How we run it: MLflow

We take the metric definitions from RAGAS and run them as judges in MLflow, which follow the same principles. MLflow does three jobs for us [8]:

- **Datasets.** An evaluation dataset is a set of records with inputs and optional expected answers. Records can be written by hand or added from logged traces.
- **Offline evaluation.** A run applies the judges to the dataset and stores the scores, so two versions of the pipeline can be compared question by question.
- **Online metrics.** The same judges score production traces, so offline and online numbers come from the same definitions.

MLflow is one of several tools for these jobs. LangSmith covers them for teams built on LangChain and LangGraph, Langfuse is a common open-source self-hosted choice, Arize Phoenix is built around OpenTelemetry traces, and Braintrust is a closed platform centred on evaluation [6].

How do you evaluate your RAG system: do you trust an LLM judge, and how did you calibrate it?

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

1. E. Mukans, G. Barzdins. Engineering a production RAG-based agent platform. In *Joint Proceedings of Baltic DB&IS Conference Forum and Doctoral Consortium 2026*, CEUR Workshop Proceedings, vol. 4211, pp. 43–54, 2026. [CEUR-WS](https://ceur-ws.org/Vol-4211/paper05.pdf)
2. J. Chen, H. Lin, X. Han, L. Sun. Benchmarking large language models in retrieval-augmented generation. In *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(16):17754–17762, 2024. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/29728)
3. S. Es, J. James, L. Espinosa Anke, S. Schockaert. RAGAs: Automated evaluation of retrieval augmented generation. In *Proceedings of EACL 2024: System Demonstrations*, pp. 150–158. ACL, 2024. [doi:10.18653/v1/2024.eacl-demo.16](https://doi.org/10.18653/v1/2024.eacl-demo.16)
4. Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, C. Zhu. G-Eval: NLG evaluation using GPT-4 with better human alignment. In *Proceedings of EMNLP 2023*, pp. 2511–2522. ACL, 2023. [doi:10.18653/v1/2023.emnlp-main.153](https://doi.org/10.18653/v1/2023.emnlp-main.153)
5. L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, I. Stoica. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In *Advances in Neural Information Processing Systems 36 (Datasets and Benchmarks Track)*, 2023. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract.html)
6. M. Grace, J. Hadfield, R. Olivares, J. De Jonghe. Demystifying evals for AI agents. Anthropic Engineering blog, 9 January 2026. [anthropic.com](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
7. A. McNamara. Building production-ready agentic systems: Lessons from Shopify Sidekick. Shopify Engineering blog, 26 August 2025. [shopify.engineering](https://shopify.engineering/building-production-ready-agentic-systems)
8. MLflow. Evaluating LLMs and agents with MLflow. Documentation, accessed November 2026. [mlflow.org](https://mlflow.org/docs/latest/genai/eval-monitor/)
