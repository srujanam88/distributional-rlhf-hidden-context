# DPL Hidden-Context RLHF

This repository contains synthetic RLHF experiments demonstrating a **failure mode of standard preference aggregation under hidden context**, and how **Distributional Preference Learning (DPL)** mitigates it.

## Overview
Constructed a controlled preference-learning environment where annotators differ along **culture, income level, political views, and risk preferences**. Although all annotators evaluate the same response pairs, their latent utilities differ due to hidden context (e.g., different safety thresholds or tolerance for risk).

Standard RLHF learns a single reward function by averaging preferences, which can perform well for some groups while causing **high regret and severe safety violations** for others.  
DPL instead models a **distribution over utilities**, enabling risk-sensitive decision rules that improve robustness and fairness.

## Key Results
- Standard RLHF exhibits high worst-case regret and near-certain safety violations for some cultural groups.
- DPL with risk-averse (quantile-based) decision rules reduces worst-case regret and eliminates safety violations across groups.
- Modeling preference uncertainty is critical for alignment under hidden context.

## Contents
- Synthetic data generation with hidden annotator context
- Standard RLHF reward model
- DPL mean–variance and categorical reward models
- Group-wise evaluation of regret, safety violations, and fairness

## Reference
Inspired by:  
**Distributional Preference Learning: Understanding and Accounting for Hidden Context in RLHF**
