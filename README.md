# Evaluation of LLMs for Mathematical Problem Solving

This repository presents a structured evaluation framework for assessing and comparing the mathematical problem-solving abilities of state-of-the-art Large Language Models (LLMs). The project focuses on model accuracy, reasoning quality, and common shortcomings across multiple datasets with different levels of mathematical difficulty.

## 🎯 Project Goals

- Benchmark LLMs such as Gemini-2.0, ChatGPT-4o, and DeepSeek-V3 on mathematical reasoning datasets.
- Compare model performance across tasks ranging from grade-school arithmetic to university-level mathematical problems.
- Identify strengths and weaknesses in model outputs, such as missing intermediate steps, weak justification, imprecise reasoning, and notation issues.
- Support both quantitative and qualitative analysis of LLMs for mathematical problem solving.
  
## 📁 Repository Structure

```text
LLMs-mathematics/
├── code/             # Evaluation scripts for each model
│   ├── DeepSeek-V3/
│   ├── GPT-4o/
│   └── Gemini-2.0/
├── datasets/         # Benchmark and university-level datasets
│   ├── university_level/
│   ├── gsm8k.jsonl
│   └── math500.jsonl
├── result/           # Model outputs and evaluation results
│   ├── DeepSeek/
│   ├── GPT-4o/
│   └── Gemini-2.0/
└── README.md         # Project overview and usage guide
```

## 📦 Datasets

This repository currently evaluates models on three datasets:
- GSM8K: Grade-school math word problems, focusing on multi-step arithmetic and basic reasoning.
- MATH500: A collection of problems from domains such as Algebra, Geometry, Number Theory, Precalculus, and Counting & Probability.
- MIT Dataset: University-level mathematical problems download from MIT OpenCourse Ware, covering advanced topics such as optimisation, statistics, and related mathematical subjects.

Each dataset includes:
- Problem statements (with LaTeX or readable formatting)
- Correct human-written solutions
- AI-generated answers from each model

## 🧪 Evaluation Metrics

The framework supports both quantitative and qualitative evaluation, including:
- Final answer accuracy
- Reasoning quality
- Completeness of solutions
- Common shortcoming analysis, such as:
   •	missing intermediate steps
	•	lack of justification
	•	imprecise reasoning
	•	unclear notation
	•	weak logical flow


Visual tools include:
- Histograms of model accuracy
- Boxplots of scoring across subjects
- Heatmaps of shortcomings correlated with subject areas

The repository is intended to support comparative analysis of how different LLMs perform on mathematical reasoning tasks.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended: a virtual environment or conda environment
- Valid API access for the corresponding models

### Installation

```bash
git clone https://github.com/sydney-machine-learning/LLM-mathematics.git
cd LLM-mathematics
```

## 🛠️ Usage Instructions

The evaluation scripts are organised by model.

### GPT-4o

   ```bash
   cd code/GPT-4o
   python gpt_gsm8k_paper_version.py
   python gpt_math500_paper_version.py
   python gpt_university_paper_version.py
   ```
### DeepSeek-V3
 ```bash
   cd code/DeepSeek-V3
   python deepseek_gsm8k_paper_version.py
   python deepseek_math500_paper_version.py
   python deepseek_university_paper_version.py
 ```

### Gemini-2.0

   ```bash
   cd code/Gemini-2.0
   python gemini_gsm8k_paper_version.py
   python gemini_math500_paper_version.py
   python gemini_university_paper_version.py
   ```
Please make sure the required packages are installed and the corresponding API keys are configured before running the scripts.


## 📈 Output and Result Analysis

Examples include:
- result/GPT-4o/
- result/DeepSeek/
- result/Gemini-2.0/

These files can be used for:
- result checking
- comparative model analysis
- qualitative error analysis
- research reporting and visualisation


## Contributing

This repository is mainly intended to support the experiments and documentation for the associated paper. If you find issues or have suggestions, please contact the authors or open an issue in the repository.


## 📌 Notes
- Scripts are currently organised by model rather than through a single unified entry script.
- Output files may vary slightly across models and datasets.
