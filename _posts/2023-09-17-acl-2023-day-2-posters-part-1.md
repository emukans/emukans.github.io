---
layout: post
title: ACL 2023 Day 2 (Posters)
date: 2023-09-17 15:19 +0300
img_path: /assets/post/acl2023-2
categories: [Notes, Posters]
tags: [acl2023]
---

## DarkBERT: A Language Model for the Dark Side of the Internet

### Motivation
Most of pretrained models are trained on the "Surface Web" data, i.e., the content that is readily available and indexed in common search engines such as Google.
Therefore the application of these models is limited in cybersecurity and "underground" domain.

### Method
![DarkBERT pretraining](darkbert-workflow.png)
The authors collected a new dataset by crawling Tor network. Then they classified the data using CoDA dataset, removed duplicates and balanced by the categories.

### Results
![DarkBERT pretraining](darkbert-results.png)
The pretrained model outperforms vanilla BERT models on the "dark" domain of tasks. 

There're many application of the model, e.g., activity classification, phishing & ransomware site detection, noteworthy thread detection and threat keyword inference

The model is available on [HuggingFace](https://huggingface.co/s2w-ai/DarkBERT). The use-case data is available upon request.


## LENS: A Learnable Evaluation Metric for Text Simplification
### Motivation
![LENS scoring sample](lens-sample.png)
There are no solid metric for text simplification task. Most of generic metrics, such as BLEU, BERTScore and specific metrics like SARI correlates with human judgement very poorly.
The authors introduce a *learnable* metric, that demonstrates high correlation with the human scoring. 

### Method
The authors fine-tuned RoBERTa model on their own dataset (SimpEval_past) and then prepare a new benchmark for evaluation (SimpEval2022) text simplification.

### Results
The new method shows higher correlation with human judgement not only on their own benchmark, but also on foreign benchmarks.
The data as well as the code is available [here](https://github.com/Yao-Dou/LENS).


## WebIE: Faithful and Robust Information Extraction on the Web
### Motivation
Existing Information Enxtraction (IE) datasets mostly based on Wikipedia and not applicable to general web domain where text is noisy and not containing fact triples.
Generative models trained on those datasets tend to hallucinate and produce a high rate of false positive results.

### Method
The authors present [WebIE](https://github.com/amazon-science/WebIE), entity-linked multilingual IE dataset collected from [C4](https://huggingface.co/datasets/c4) dataset.
Part of the dataset is manually annotated and include a negative examples to better reflect the data on the web.

![WebIE Statistics](webie-statistics.png)

### Results
Models trained on WebIE are more generalisable:
* Models trained only on Wikipedia-domain IE datasets leads to 0% accuracy on negative examples
* Models trained on WebIE show promising zero-shot performance on Wikipedia-domain IE datasets
* Models trained on both datasets show best results

![WebIE results](webie-results.png)

## UniSumm and SummZoo: Unified Model and Diverse Benchmark for Few-Shot Summarization
### Motivation
High annotation costs motivate the research of few-shot summarization, which aims to build specific summarization scenario using very few ground-truth data.
However, development of few-shot summarization research is limited from less satisfying summarization models and miscellaneous evaluation benchmark.

### Method
UniSumm, a model that re-uses existing data and can be easily adapted to diverse tasks.


![UniSumm method](unisumm-method.png)

SummZoo, a diverse benchmark consisting of 8 summarization tasks.


### Results
