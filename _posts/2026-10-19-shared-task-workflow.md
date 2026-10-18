---
title: "A 10-Step Workflow for NLP Shared Tasks on a Small Compute Budget"
description: "How to spend limited compute in an NLP shared task: a three-phase, 10-step workflow from my doctoral thesis."
date: 2026-10-19 00:00:00 +0300
categories: [Research, Shared tasks]
tags: [shared-tasks, semeval, smm4h, fine-tuning, methodology]
image:
  path: /assets/img/posts/shared-task-workflow/cover.png
  alt: "A 10-step workflow in three phases: get a working pipeline, beat a fixed baseline with new features, then scale the model and simplify. Step 10 loops back to step 7 with external knowledge."
---

I have recently submitted my doctoral thesis, *Application of Large Language Models for Input Context Enrichment in NLP Pipelines*, for review. This post is adapted from one of its sections: the development workflow we used across four SemEval and SMM4H shared tasks between 2022 and 2025, together with my supervisor Guntis Barzdiņš.

A shared task gives you a fixed dataset, a fixed metric and a hard deadline. With a small compute budget, the question that matters is how to spend your GPU hours so that few of them go into dead-end experiments. The workflow below is our answer: 10 steps in three phases. Step 10 loops back to Step 7: when the training data has no more features to give, bring in external knowledge.

## Phase I: get a working pipeline (Steps 1–5)

The goal is an end-to-end pipeline that runs. Performance does not matter yet.

1. **Start with the smallest model.** Take the base variant, not the large one. A small model surfaces pipeline bugs as well as a big one does, at a fraction of the cost.
2. **Train on the development subset first.** A full cycle should take minutes, not hours.
3. **Time-box the first runs** to a few epochs or a few minutes. Check that metrics are logged correctly and that the output is not degenerate.
4. **Train on plain, unmodified data.** This is the sanity check before any feature work.
5. **Read the outputs yourself.** Automatic metrics can miss real progress, or reward a model that repeats frequent phrases. We ran into this in our first task ([2022](https://aclanthology.org/2022.semeval-1.9/)).

Parameter-efficient methods such as adapters suit this phase well: they are fast and stable. Just don't expect them to give your best final score ([2023](https://aclanthology.org/2023.semeval-1.45/)).

## Phase II: develop features (Steps 6–7)

6. **Train a baseline on the full training set.** Its score is the reference for every later experiment.
7. **Add features one at a time**, and test each at small scale first. If a feature does not move the metric but the outputs look better on inspection, keep it for a larger test.

## Phase III: scale and refine (Steps 8–10)

8. **Scale the model** once the features work. Scaling can absorb some features completely.
9. **Drop redundant features.** If two configurations score the same, keep the simpler one.
10. **Repeat the feature cycle at the new scale.** When the training data has no more features to give, bring in external knowledge: LLM-generated context about the input ([2023](https://aclanthology.org/2023.semeval-1.45/), [2024](https://aclanthology.org/2024.smm4h-1.6/)) or a curated database ([2025](https://workshop-proceedings.icwsm.org/abstract.php?id=2025_66)).

For the final runs, train the full model and don't rely on early stopping. In our experiments the task metric kept improving well after the validation loss had stopped ([2022](https://aclanthology.org/2022.semeval-1.9/), [2023](https://aclanthology.org/2023.semeval-1.45/)).

## Papers and code

The details of each task, with results and ablations, are in the original papers:

- **2022** · *RIGA at SemEval-2022 Task 1: Scaling Recurrent Neural Networks for CODWOE Dictionary Modeling.* [Paper](https://aclanthology.org/2022.semeval-1.9/) · [Code](https://github.com/emukans/codwoe)
- **2023** · *RIGA at SemEval-2023 Task 2: NER Enhanced with GPT-3.* [Paper](https://aclanthology.org/2023.semeval-1.45/) · [Code](https://github.com/emukans/multiconer2-riga)
- **2024** · *RIGA at SMM4H-2024 Task 1: Enhancing ADE Discovery with GPT-4.* [Paper](https://aclanthology.org/2024.smm4h-1.6/) · [Code](https://github.com/emukans/smm4h2024-riga)
- **2025** · *RIGA at SMM4H-HeaRD 2025: Context-Enriched Classification Pipeline.* [Paper](https://workshop-proceedings.icwsm.org/abstract.php?id=2025_66) · [Code](https://github.com/emukans/smm4h2025-riga)
