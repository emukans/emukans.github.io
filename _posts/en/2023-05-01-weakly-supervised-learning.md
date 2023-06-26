---
layout: post
title: Weakly supervised learning
date: 2023-05-01 19:51 +0300
img_path: /assets/post/weakly-supervised
categories: [Tutorial]
tags: [weak supervision, unsupervised]
---

Machine learning has revolutionized the way we solve complex problems, from natural language processing to image recognition.
However, one major obstacle that machine learning practitioners face is the high cost of data labeling,
which is often necessary to train models effectively.
Fortunately, weak supervision has emerged as a powerful solution to this problem.
In this article, we will explore the three types of weak supervision and the techniques used in each of them.

![Unlabeled data](weakly supervided learning.jpg)
_An example of the influence of unlabeled data in weakly supervised learning. [Source](https://en.wikipedia.org/wiki/Weak_supervision)_

The three types of weak supervision:
* Incomplete supervision - only a small subset of training data are given with labels whereas the other data remain unlabeled;
* Inexact supervision - only coarse-grained labels are given;
* Inaccurate supervision - the given labels are not always ground-truth.

Let's dive deeper into each type of supervision and the techniques used to tackle them.

## Incomplete supervision

Active learning and semi-supervised learning are two major techniques used in incomplete supervision.
Active learning assumes that there is an "oracle," such as a human expert, that can be queried to obtain ground-truth labels.
On the other hand, semi-supervised learning attempts to exploit unlabeled data in addition to labeled data to improve learning performance,
without any human intervention.

The main goal in active learning is to minimize the number of queries to reduce the cost of training.
This issue could be approached by trying to selects the most valuable unlabeled instance to query using two criteria:
informativeness and representativeness.
Informativeness refers to how well an unlabeled instance helps reduce the uncertainty of a statistical model,
while representativeness refers to how well an instance helps represent the structure of input patterns.

In semi-supervised learning, no human expert is involved, and the algorithm tries to explore the data using unsupervised methods
like cluster and manifold assumptions. Both assumptions believe that similar data points should have similar outputs. 

Somewhere in between these two methods, there is another one which mixes both approaches.
There are labeling functions that are given by experts. These functions will cover some parts of the data points.
Using these labeled data points, we can train a probabilistic model to label the other uncovered cases.
Solutions like [Snorkel](https://snorkel.ai/) by Stanford, [skweak](https://spacy.io/universe/project/skweak) for NLP and [ASTRA](https://github.com/microsoft/ASTRA) by Microsoft uses that approach.

It is worth mentioning that although the learning performance is expected to be improved by exploiting unlabeled data,
in some cases the performance may become worse after semi-supervised learning.
The exploitation of unlabeled data naturally leads to more than one model option, and inadequate choice may lead to poor performance.
The fundamental strategy to make semi-supervised learning "safer" is to optimize the worst-case performance among the options,
possibly by incorporating ensemble mechanisms.

## Inexact supervision

Multi-instance learning is the main approach used in inexact supervision.
In multi-instance learning, a set of data points (bag) is positive if some subset of a bag is also positive.
The goal of multi-instance learning is to predict labels for unseen bags.
This type of supervision is commonly used in applications like drug discovery and medical image analysis.

## Inaccurate supervision

The basic idea of inaccurate supervision is to identify potentially mislabeled examples and make corrections.
This can be achieved using voting strategies or clustering methods to find outliers.
By identifying and correcting mislabeled examples, we can improve the quality of the training data and, consequently,
the accuracy of the models.

## Conclusion
Weak supervision has emerged as a powerful solution to the high cost of data labeling. In practice,
the solution usually involves a mixture of all three types of supervision.
