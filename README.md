
# Evaluation of LLMs for Mathematical Problem Solving

This repository presents a structured evaluation framework designed to assess and compare the mathematical problem-solving abilities of state-of-the-art Large Language Models (LLMs). The evaluation focuses on accuracy, reasoning quality, and error patterns across multiple datasets representing diverse mathematical domains.

## 🎯 Project Goals

- **Benchmark LLMs** such as Gemini-2.0, ChatGPT-4o, and DeepSeek-V3 on challenging math datasets.
- **Identify strengths and weaknesses** in the reasoning chains of LLM outputs using human-verified solutions.
- **Categorize common error types**, such as logical flow issues, misinterpretation of problem conditions, and incomplete solutions.
- **Visualize performance** using statistical graphs and heatmaps to facilitate comparative analysis.

## 📁 Repository Structure

```bash
Evaluation-of-LLMs-for-mathematical-problem-solving/
├── code/             # Scripts for evaluation, analysis, and visualization
├── datasets/         # Curated mathematical datasets with ground truth
├── result/           # Accuracy scores, error classifications, and charts
├── figures/          # Generated visualizations (e.g., histograms, pie charts)
├── requirements.txt  # Python dependencies
└── README.md         # Project overview and usage guide
```

## 📦 Datasets

This repository evaluates models on three datasets:
- **GSM8K**: Grade-school math word problems, focusing on step-by-step arithmetic and logic.
- **Math500**: A collection of problems from Algebra, Geometry, Number Theory, etc., grouped by difficulty levels.
- **UNSW Problems**: University-level questions used in actual exams across topics like Optimization, Statistics, and Computational Finance.

Each dataset includes:
- Problem statements (with LaTeX or readable formatting)
- Correct human-written solutions
- AI-generated answers from each model

## 🧪 Evaluation Metrics

The framework supports both **quantitative and qualitative** evaluation:

- **Accuracy** (% of exactly correct answers)
- **Chain-of-thought analysis** (quality of step-by-step reasoning)
- **Shortcoming classification** (e.g., arithmetic mistake, misinterpretation, lack of logic)

Visual tools include:
- Histograms of model accuracy
- Boxplots of scoring across subjects
- Heatmaps of shortcomings correlated with subject areas

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Recommended: virtualenv or conda environment

### Installation

```bash
git clone https://github.com/SHARP-Mitsuko/Evaluation-of-LLMs-for-mathematical-problem-solving.git
cd Evaluation-of-LLMs-for-mathematical-problem-solving
pip install -r requirements.txt
```

## 🛠️ Usage Instructions

1. Navigate to the code directory:
   ```bash
   cd code
   ```

2. Run the model evaluation and scoring:
   ```bash
   python evaluate_models.py
   ```

3. Generate visual analysis:
   ```bash
   python generate_figures.py
   ```

4. To update or add new AI responses, edit the respective CSV or JSONL files under `datasets/`.

## 📈 Output and Result Analysis

- **`result/accuracy_summary.csv`**: Contains model-wise accuracy on each dataset.
- **`result/shortcomings_by_subject.csv`**: Shows error breakdown by type and topic.
- **`figures/`**: Includes exported `.png` files for histograms, pie charts, heatmaps, etc.

Use these files for:
- Paper figures or presentations
- Comparative model assessment
- Insights into LLM failure modes in math reasoning

## 🤝 Contributing

We welcome contributions to improve model evaluation, expand datasets, or refine error taxonomies.

To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b improve-eval`.
3. Make changes and commit: `git commit -am 'Enhance evaluation process'`.
4. Push to the branch: `git push origin improve-eval`.
5. Submit a pull request.
