import json
import openai
import os
import re
import time
import sympy as sp
from pathlib import Path
from collections import defaultdict

# =========================
# Basic configuration
# =========================
API_KEY = os.getenv("OPENAI_API_KEY", "")
client = openai.Client(api_key=API_KEY)

DATA_PATH = "datasets/math500.jsonl"
OUTPUT_DIR = Path("results/gpt4o_math500")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "math500_gpt4o_results.jsonl"
FAILED_PATH = OUTPUT_DIR / "math500_gpt4o_failed_cases.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "math500_gpt4o_summary.json"
LEVEL_STATS_PATH = OUTPUT_DIR / "math500_gpt4o_level_stats.json"
SUBJECT_STATS_PATH = OUTPUT_DIR / "math500_gpt4o_subject_stats.json"

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2
REQUEST_TIMEOUT = 30

N = 500  # evaluate first 500 items


# =========================
# Utilities
# =========================
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def balance_braces(expr):
    open_count = expr.count("{")
    close_count = expr.count("}")
    if close_count < open_count:
        expr += "}" * (open_count - close_count)
    return expr


def extract_boxed(expr):
    start = expr.find(r"\boxed{")
    if start == -1:
        return expr.strip()

    i = start + len(r"\boxed{")
    brace_count = 1
    content = ""

    while i < len(expr) and brace_count > 0:
        if expr[i] == "{":
            brace_count += 1
        elif expr[i] == "}":
            brace_count -= 1
        content += expr[i]
        i += 1

    if brace_count != 0:
        return "❌ Invalid: Unmatched braces"

    content = content[:-1] if content.endswith("}") else content
    content = balance_braces(content)

    if re.fullmatch(r"\\frac\{[^{}]+\}", content):
        return "❌ Invalid: Incomplete \\frac"

    return content.strip()


def clean_latex(expr):
    if expr is None:
        return ""

    expr = str(expr)
    expr = expr.replace(" ", "").replace("\\left", "").replace("\\right", "")
    expr = expr.replace("^\\circ", "").replace("\\\\", "\\")
    expr = re.sub(r"\\sqrt([a-zA-Z0-9])", r"\\sqrt{\1}", expr)
    expr = re.sub(r"\\frac([0-9])([0-9])", r"\\frac{\1}{\2}", expr)
    return expr


def is_valid_latex_fraction(expr):
    return not bool(re.search(r"\\frac\{[^{}]+\}(?!\{)", expr))


def is_equivalent(expr1, expr2):
    expr1 = clean_latex(expr1)
    expr2 = clean_latex(expr2)

    if "\\text" in expr1 or "\\text" in expr2:
        return expr1 == expr2

    try:
        a = sp.simplify(sp.sympify(expr1))
        b = sp.simplify(sp.sympify(expr2))
        return sp.simplify(a - b) == 0
    except Exception:
        return expr1 == expr2


# =========================
# GPT call
# =========================
def solve_problem(prompt):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=100,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a rigorous math solver.

Your task:
- Solve the math problem below.
- Only return the final answer in LaTeX, wrapped inside \\boxed{...}.
- DO NOT output any explanation, steps, or reasoning.
- If you cannot solve the problem, return \\boxed{?}.
- Your output must be a single valid \\boxed{...} expression — nothing else.

Example of valid output: \\boxed{\\frac{1}{2}}"""
                    },
                    {"role": "user", "content": prompt}
                ],
                timeout=REQUEST_TIMEOUT
            )

            raw = response.choices[0].message.content.strip()

            return {
                "success": True,
                "raw_output": raw,
                "parsed_answer": extract_boxed(raw),
                "error_message": None,
                "attempt_used": attempt
            }

        except Exception as e:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                return {
                    "success": False,
                    "raw_output": None,
                    "parsed_answer": None,
                    "error_message": str(e),
                    "attempt_used": attempt
                }


# =========================
# Main evaluation
# =========================
data = load_jsonl(DATA_PATH)
N = min(N, len(data))

correct = 0
evaluated = 0

level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

results = []
failed_cases = []

print("\n====== ❌ Incorrect Answers ======\n")

for i in range(N):
    item = data[i]
    q = item["problem"]
    ref = clean_latex(item["answer"])
    level = item.get("level", "unknown")
    subject = item.get("subject", "unknown")

    result = solve_problem(q)

    if not result["success"]:
        failed_cases.append({
            "index": i + 1,
            "question": q,
            "level": level,
            "subject": subject,
            "error_message": result["error_message"],
            "attempt_used": result["attempt_used"]
        })
        continue

    pred_raw = result["parsed_answer"]
    pred = clean_latex(pred_raw)

    if "❌" in pred_raw or not is_valid_latex_fraction(pred):
        failed_cases.append({
            "index": i + 1,
            "question": q,
            "level": level,
            "subject": subject,
            "error_message": "Invalid LaTeX output",
            "raw_output": result["raw_output"],
            "parsed_answer": pred_raw
        })
        continue

    evaluated += 1
    level_stats[level]["total"] += 1
    subject_stats[subject]["total"] += 1

    is_correct = is_equivalent(pred, ref)

    if is_correct:
        correct += 1
        level_stats[level]["correct"] += 1
        subject_stats[subject]["correct"] += 1
    else:
        print(f"❌ Question {i+1} incorrect")
        print(f"✅ Correct answer: {ref}")
        print(f"❌ GPT generated: {pred}\n")

    results.append({
        "index": i + 1,
        "question": q,
        "correct_answer_raw": item["answer"],
        "correct_answer_clean": ref,
        "pred_raw_output": result["raw_output"],
        "pred_parsed": pred_raw,
        "pred_clean": pred,
        "level": level,
        "subject": subject,
        "is_correct": is_correct,
        "attempt_used": result["attempt_used"]
    })

# =========================
# Summary
# =========================
accuracy = 100 * correct / evaluated if evaluated > 0 else 0.0

summary = {
    "model": "gpt-4o",
    "dataset": "MATH500",
    "total_requested": N,
    "evaluated": evaluated,
    "correct": correct,
    "accuracy_percent": round(accuracy, 2),
    "failed_cases": len(failed_cases),
    "max_retries": MAX_RETRIES,
    "timeout_seconds": REQUEST_TIMEOUT
}

print(f"✅ Total correct: {correct} / {evaluated}")
print(f"✅ GPT-4o total accuracy: {accuracy:.2f}%\n")

# =========================
# Print category accuracy
# =========================
def print_accuracy(title, stats):
    print(f"📊 {title} Accuracy:")
    for key, v in sorted(stats.items()):
        acc = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0.0
        print(f" - {key}: {acc:.2f}%")
    print()

print_accuracy("Level-wise", level_stats)
print_accuracy("Subject-wise", subject_stats)

# =========================
# Save outputs
# =========================
save_jsonl(RESULTS_PATH, results)
save_jsonl(FAILED_PATH, failed_cases)

with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

with open(LEVEL_STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(level_stats, f, ensure_ascii=False, indent=2)

with open(SUBJECT_STATS_PATH, "w", encoding="utf-8") as f:
    json.dump(subject_stats, f, ensure_ascii=False, indent=2)
