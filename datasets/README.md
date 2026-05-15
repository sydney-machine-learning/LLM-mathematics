# Dataset Description

This folder contains the datasets and source documents used in the project **Evaluation of LLMs for Mathematical Problem Solving**.

## 1. GSM8K

GSM8K is a benchmark dataset for grade-school mathematical reasoning. This repository includes the GSM8K test set used in our experiments for reference and reproducibility.

The evaluation mainly focuses on final-answer accuracy, where model outputs are compared with the reference answers.

## 2. MATH500

The MATH500 dataset contains 500 mathematical problems sampled from the MATH dataset.

In this project, MATH500 is used to evaluate model performance on more challenging mathematical reasoning tasks. The problems cover different mathematical subjects and difficulty levels. The evaluation mainly focuses on final-answer accuracy, with model outputs compared against the reference answers.

## 3. University-Level Source Documents

The university-level problem materials are stored as PDF source documents.

These PDF files were downloaded from MIT OpenCourse Ware and were used to support the construction of the university-level mathematics evaluation set. The documents include problems from different university-level mathematics areas, such as financial mathematics, optimisation, and statistics.

At the current stage, these university-level materials are provided as source/problem documents rather than a fully structured CSV or JSON dataset. Some related evaluation scripts may expect a future structured file, such as:

```text
university_level_subquestions.csv
