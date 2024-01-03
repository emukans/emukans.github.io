---
layout: post
title: Inductive bias
date: 2023-07-23 14:09 +0300
---

Inductive bias in machine learning refers to the set of assumptions or constraints embedded within an algorithm that guide its learning process.
Generally, every building block and every belief that we make about the data is a form of inductive bias.

Let's break down the most common architectures and set a level of inductive bias (IB) it has.

## Regression models
IB: Strong

Amount of data: Small

Form of the bias: The data is constrained to a single family of equations and optimizes only the coefficients.

## Decision trees
IB: Strong

Amount of data: Small

Form of the bias: The data is constrained by the assumption, that the objective can be achieved by asking a series of binary questions.

## Bayesian models


