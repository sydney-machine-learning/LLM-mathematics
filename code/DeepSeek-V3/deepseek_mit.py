import os
import re
import json
import time
import math
import requests
import pandas as pd
from typing import Any, Dict, List
from dotenv import load_dotenv

load_dotenv()

# ======================
# Basic configuration
# ======================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"


DEEPSEEK_MODEL = "deepseek-chat"

INPUT_CSV = "mit_ocw_subquestions.csv"
SOLUTIONS_JSON = "deepseek_v3_subquestion_solutions.json"
GRADING_JSON = "deepseek_v3_self_grading_results.json"
WRONG_JSON = "deepseek_v3_wrong_subquestions.json"
SUMMARY_CSV = "deepseek_v3_score_distribution_by_category.csv"
ERROR_LOG = "deepseek_v3_api_errors.log"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_RETRIES = 2
TEMPERATURE = 0.0

if not DEEPSEEK_API_KEY:
    raise ValueError("未检测到 DEEPSEEK_API_KEY 环境变量，请先在 .env 或系统环境变量中设置。")


# ======================
# Helpers
# ======================
def clean_nan(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    elif isinstance(obj, list):
        return [clean_nan(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    return obj


def normalize_id(value: Any, default: str = "None") -> str:
    if pd.isna(value):
        return default
    s = str(value).strip()
    if s.lower() == "nan" or s == "":
        return default
    return s


def clean_json_block(text: str) -> str:
    if not text:
        return ""

    text = text.strip()

    if text.startswith("```json"):
        text = text[len("```json"):].strip()
    elif text.startswith("```"):
        text = text[len("```"):].strip()

    if text.endswith("```"):
        text = text[:-3].strip()

    return text


def log_error(message: str) -> None:
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def load_input_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_cols = ["qid", "subid", "subquestion", "full_question"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    df["qid"] = df["qid"].apply(lambda x: normalize_id(x, "UnknownQ"))
    df["subid"] = df["subid"].apply(lambda x: normalize_id(x, "None"))
    df["subquestion"] = df["subquestion"].fillna("").astype(str).str.strip()
    df["full_question"] = df["full_question"].fillna("").astype(str).str.strip()

    if "subject" in df.columns:
        df["subject"] = df["subject"].fillna("Unknown").astype(str).str.strip()
    else:
        df["subject"] = "Unknown"

    return df


# ======================
# Prompt builders
# ======================
def build_solution_prompt(full_question: str, subquestion: str) -> str:
    return f"""You are a mathematics and statistics master's student.

Here is the full background of the problem:
{full_question}

Now solve the following sub-question step by step using structured reasoning:

{subquestion}

Return the solution as a JSON array.
Each item in the array should have:
- "step": the step number, or "final" for the final answer
- "desc": a short description of the step
- "expr": the mathematical expression used, if applicable
- "value": the computed result, if applicable

Return ONLY a valid JSON array.
Do not use markdown.
Do not include any extra explanation outside the JSON array.
"""


def build_grading_prompt(subquestion: str, steps_json: List[Dict[str, Any]]) -> str:
    steps_str = json.dumps(steps_json, indent=2, ensure_ascii=False)

    return f"""You are a mathematics tutor. Please grade a student's step-by-step solution to a sub-question.

Sub-question:
{subquestion}

Student's steps:
{steps_str}

Now do the following:
1. Evaluate whether each step is correct or flawed. If flawed, explain why.
2. Give a short comment for each step.
3. Give an overall score out of 5 and a short feedback.
4. Use this scoring rubric:
   1 - Completely incorrect: Major logical flaws, fundamental misunderstandings, or missing core steps.
   2 - Weak: Some grasp of the method, but contains multiple errors, flawed reasoning, or incoherent structure.
   3 - Satisfactory: Main method is correct, includes key steps, but has some calculation or explanation issues.
   4 - Good: Mostly correct, logically structured, only minor issues such as small errors or slightly informal reasoning.
   5 - Excellent: Fully correct, well-organized, rigorous and clear reasoning. A model solution.

Return ONLY a valid JSON object in the format:

{{
  "score": X,
  "total": 5,
  "feedback": "...",
  "step_feedback": [
    {{"step": "...", "comment": "..."}}
  ]
}}

Do not use markdown.
Do not include any extra explanation outside the JSON object.
"""


# ======================
# API calls
# ======================
def call_deepseek_api(messages: List[Dict[str, str]], qid: str, subid: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
            }

            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            reply = response.json()["choices"][0]["message"]["content"].strip()
            return clean_json_block(reply)

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"⚠️ DeepSeek 重试 {attempt}/{MAX_RETRIES} -> {qid}-{subid}: {last_error}")
                time.sleep(SLEEP_BETWEEN_RETRIES)

    log_error(f"[{qid}-{subid}] {last_error}")
    raise RuntimeError(last_error)


def solve_with_deepseek(full_question: str, subquestion: str, qid: str, subid: str) -> List[Dict[str, Any]]:
    prompt = build_solution_prompt(full_question, subquestion)

    try:
        reply = call_deepseek_api(
            messages=[{"role": "user", "content": prompt}],
            qid=qid,
            subid=subid,
        )
        parsed = json.loads(reply)

        if not isinstance(parsed, list):
            raise ValueError("DeepSeek solution output is not a JSON array")

        return parsed

    except Exception as e:
        print(f"❌ DeepSeek 解题失败（{qid}-{subid}）：{e}")
        return [{
            "step": "error",
            "desc": "Invalid JSON output or API failure",
            "expr": None,
            "value": None,
            "raw_error": str(e)
        }]


def grade_with_deepseek(subquestion: str, steps: List[Dict[str, Any]], qid: str, subid: str) -> Dict[str, Any]:
    prompt = build_grading_prompt(subquestion, steps)

    try:
        reply = call_deepseek_api(
            messages=[{"role": "user", "content": prompt}],
            qid=qid,
            subid=subid,
        )
        parsed = json.loads(reply)

        if not isinstance(parsed, dict):
            raise ValueError("DeepSeek grading output is not a JSON object")

        parsed.setdefault("score", 0)
        parsed.setdefault("total", 5)
        parsed.setdefault("feedback", "")
        parsed.setdefault("step_feedback", [])
        return parsed

    except Exception as e:
        print(f"❌ DeepSeek 评分失败（{qid}-{subid}）：{e}")
        return {
            "score": 0,
            "total": 5,
            "feedback": f"API/JSON Error: {str(e)}",
            "step_feedback": [],
        }


# ======================
# Main pipeline
# ======================
def generate_solutions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    solutions = []

    for _, row in df.iterrows():
        qid = row["qid"]
        subid = row["subid"]
        subq = row["subquestion"]
        fullq = row["full_question"]
        subject = row["subject"]

        print(f"\n🧠 解题中：{qid}-{subid}")

        steps = solve_with_deepseek(fullq, subq, qid, subid)

        solutions.append({
            "qid": qid,
            "subid": subid,
            "subject": subject,
            "subquestion": subq,
            "full_question": fullq,
            "steps": clean_nan(steps),
        })

        time.sleep(1)

    return solutions


def grade_solutions(solutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    graded_results = []

    for idx, item in enumerate(solutions):
        qid = item["qid"]
        subid = item["subid"]
        subject = item.get("subject", "Unknown")
        question = item["subquestion"]
        steps = item["steps"]

        print(f"📝 Grading: {qid}-{subid}")
        grading = grade_with_deepseek(question, steps, qid, subid)

        graded_results.append({
            "index": idx,
            "qid": qid,
            "subid": subid,
            "subject": subject,
            "subquestion": question,
            "score": grading.get("score", 0),
            "total": grading.get("total", 5),
            "feedback": grading.get("feedback", ""),
            "step_feedback": grading.get("step_feedback", []),
        })

        time.sleep(1)

    return graded_results


def save_wrong_subquestions(graded_results: List[Dict[str, Any]]) -> None:
    wrong = [item for item in graded_results if item.get("score", 0) < 5]

    with open(WRONG_JSON, "w", encoding="utf-8") as f:
        json.dump(wrong, f, indent=2, ensure_ascii=False)

    print(f"✅ 筛选出 {len(wrong)} 道非满分子问，保存为 {WRONG_JSON}")


def summarize_scores(graded_results: List[Dict[str, Any]]) -> pd.DataFrame:
    from collections import defaultdict, Counter

    category_distribution = defaultdict(lambda: {"score_counts": Counter(), "total": 0})

    for item in graded_results:
        subject = item.get("subject", "Unknown")
        score = int(item.get("score", 0))
        category_distribution[subject]["score_counts"][score] += 1
        category_distribution[subject]["total"] += 1

    summary_data = []
    for category, data in category_distribution.items():
        total_questions = data["total"]
        full_score = 5
        accuracy = data["score_counts"].get(full_score, 0) / total_questions if total_questions > 0 else 0

        row = {
            "课程类别": category,
            "题目总数": total_questions,
            "准确率": f"{accuracy:.2%}",
        }

        for score in range(1, 6):
            count = data["score_counts"].get(score, 0)
            row[f"得分 {score} 数"] = count
            row[f"得分 {score} 占比"] = f"{(count / total_questions):.1%}" if total_questions > 0 else "0.0%"

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return df_summary


# ======================
# Run
# ======================
def main():
    df = load_input_csv(INPUT_CSV)

    solutions = generate_solutions(df)
    with open(SOLUTIONS_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_nan(solutions), f, indent=2, ensure_ascii=False)
    print(f"\n✅ 所有子问解题完毕，结果已保存为 {SOLUTIONS_JSON}")

    graded_results = grade_solutions(solutions)
    with open(GRADING_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_nan(graded_results), f, indent=2, ensure_ascii=False)
    print(f"✅ 所有子问评分完毕，结果已保存为 {GRADING_JSON}")

    save_wrong_subquestions(graded_results)

    df_summary = summarize_scores(graded_results)
    print("\n📊 Score distribution by category:")
    print(df_summary)


if __name__ == "__main__":
    main()
