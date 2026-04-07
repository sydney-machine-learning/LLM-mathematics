import os
import json
import time
import math
import pandas as pd
import requests
from typing import Any, Dict, List
from collections import defaultdict, Counter
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# ======================
# Basic configuration
# ======================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

GEMINI_MODEL = "gemini-2.0-flash"
DEEPSEEK_MODEL = "deepseek-chat"

INPUT_CSV = "mit_ocw_subquestions.csv"

SOLUTIONS_JSON = "gemini_subquestion_solutions.json"
GRADING_JSON = "gemini_grading_results.json"
WRONG_JSON = "gemini_wrong_subquestions.json"
SUMMARY_CSV = "gemini_score_distribution_by_category.csv"
ERROR_TAXONOMY_JSON = "gemini_error_taxonomy.json"
ERROR_TAXONOMY_SUMMARY_CSV = "gemini_error_taxonomy_summary.csv"

GEMINI_ERROR_LOG = "gemini_api_errors.log"
DEEPSEEK_ERROR_LOG = "deepseek_grading_errors.log"

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_RETRIES = 2

GEMINI_TEMPERATURE = 0.0
DEEPSEEK_TEMPERATURE = 0.0

if not GEMINI_API_KEY:
    raise ValueError("未检测到 GEMINI_API_KEY 环境变量。")
if not DEEPSEEK_API_KEY:
    raise ValueError("未检测到 DEEPSEEK_API_KEY 环境变量。")

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
)


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


def log_error(path: str, message: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
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
- "value": the computed result if applicable

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


def build_taxonomy_prompt(subquestion: str, steps_json: List[Dict[str, Any]], grading: Dict[str, Any]) -> str:
    steps_str = json.dumps(steps_json, indent=2, ensure_ascii=False)
    grading_str = json.dumps(grading, indent=2, ensure_ascii=False)

    return f"""You are analysing mathematical reasoning errors in a student's response.

Sub-question:
{subquestion}

Student's steps:
{steps_str}

Grading result:
{grading_str}

Assign the response to one primary error category only.
Choose the single most dominant category from this list:

- Understanding Errors
- Deduction Errors
- Expression Errors
- Process Control Errors
- Format Errors
- No Major Error

Definitions:
- Understanding Errors: misunderstanding key concepts, quantities, symbols, assumptions, or what is being asked.
- Deduction Errors: formula derivation mistakes, invalid transformations, incorrect algebra/calculus/probability reasoning.
- Expression Errors: main idea is mostly right, but wording is unclear, notation is confusing, explanation is too vague.
- Process Control Errors: missing sub-steps, poor structure, inconsistent answers across steps, skipped conditions, incomplete chain.
- Format Errors: mostly minor presentation issues such as copied symbols wrongly, small notation slips, formatting-only issues.
- No Major Error: fully correct or no dominant error pattern.

Return ONLY a valid JSON object in the format:

{{
  "primary_error_type": "...",
  "reason": "..."
}}

Do not use markdown.
Do not include any extra explanation outside the JSON object.
"""


# ======================
# Gemini generation
# ======================
def call_gemini_api(prompt: str, qid: str, subid: str) -> List[Dict[str, Any]]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=GEMINI_TEMPERATURE,
                    max_output_tokens=2048,
                ),
            )

            raw_text = response.text.strip() if getattr(response, "text", None) else ""
            raw_text = clean_json_block(raw_text)
            parsed = json.loads(raw_text)

            if not isinstance(parsed, list):
                raise ValueError("Gemini output is not a JSON array")

            return parsed

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"⚠️ Gemini 重试 {attempt}/{MAX_RETRIES} -> {qid}-{subid}: {last_error}")
                time.sleep(SLEEP_BETWEEN_RETRIES)

    log_error(GEMINI_ERROR_LOG, f"[{qid}-{subid}] {last_error}")
    print(f"❌ Gemini 解题失败（{qid}-{subid}）：{last_error}")
    return [{
        "step": "error",
        "desc": "Invalid JSON output or API failure",
        "expr": None,
        "value": None,
        "raw_error": last_error
    }]


# ======================
# DeepSeek grading / taxonomy
# ======================
def call_deepseek_json(prompt: str, qid: str, subid: str) -> Dict[str, Any]:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": DEEPSEEK_TEMPERATURE,
            }

            response = requests.post(
                DEEPSEEK_API_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()

            reply = response.json()["choices"][0]["message"]["content"].strip()
            reply = clean_json_block(reply)
            parsed = json.loads(reply)

            if not isinstance(parsed, dict):
                raise ValueError("DeepSeek output is not a JSON object")

            return parsed

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                print(f"⚠️ DeepSeek 重试 {attempt}/{MAX_RETRIES} -> {qid}-{subid}: {last_error}")
                time.sleep(SLEEP_BETWEEN_RETRIES)

    log_error(DEEPSEEK_ERROR_LOG, f"[{qid}-{subid}] {last_error}")
    raise RuntimeError(last_error)


def grade_with_deepseek(subquestion: str, steps: List[Dict[str, Any]], qid: str, subid: str) -> Dict[str, Any]:
    prompt = build_grading_prompt(subquestion, steps)

    try:
        parsed = call_deepseek_json(prompt, qid, subid)
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


def classify_error_taxonomy(subquestion: str, steps: List[Dict[str, Any]], grading: Dict[str, Any], qid: str, subid: str) -> Dict[str, Any]:
    # 满分默认不归入错误类
    if grading.get("score", 0) >= grading.get("total", 5):
        return {
            "primary_error_type": "No Major Error",
            "reason": "Full score under rubric."
        }

    prompt = build_taxonomy_prompt(subquestion, steps, grading)

    try:
        parsed = call_deepseek_json(prompt, qid, subid)
        parsed.setdefault("primary_error_type", "No Major Error")
        parsed.setdefault("reason", "")
        return parsed

    except Exception as e:
        print(f"❌ Error taxonomy 分类失败（{qid}-{subid}）：{e}")
        return {
            "primary_error_type": "Unclassified",
            "reason": str(e)
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

        prompt = build_solution_prompt(fullq, subq)
        steps = call_gemini_api(prompt, qid, subid)

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
        taxonomy = classify_error_taxonomy(question, steps, grading, qid, subid)

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
            "primary_error_type": taxonomy.get("primary_error_type", "Unclassified"),
            "error_reason": taxonomy.get("reason", ""),
        })

        time.sleep(1)

    return graded_results


def save_wrong_subquestions(graded_results: List[Dict[str, Any]]) -> None:
    wrong = [item for item in graded_results if item.get("score", 0) < item.get("total", 5)]

    with open(WRONG_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_nan(wrong), f, indent=2, ensure_ascii=False)

    print(f"✅ 筛选出 {len(wrong)} 道非满分子问，保存为 {WRONG_JSON}")


def summarize_scores(graded_results: List[Dict[str, Any]]) -> pd.DataFrame:
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


def summarize_error_taxonomy(graded_results: List[Dict[str, Any]]) -> pd.DataFrame:
    taxonomy_distribution = defaultdict(lambda: Counter())

    for item in graded_results:
        subject = item.get("subject", "Unknown")
        error_type = item.get("primary_error_type", "Unclassified")
        taxonomy_distribution[subject][error_type] += 1

    summary_rows = []
    all_error_types = set()
    for subject, counter in taxonomy_distribution.items():
        all_error_types.update(counter.keys())

    all_error_types = sorted(all_error_types)

    for subject, counter in taxonomy_distribution.items():
        total = sum(counter.values())
        row = {
            "课程类别": subject,
            "总数": total,
        }
        for err in all_error_types:
            count = counter.get(err, 0)
            row[f"{err} 数"] = count
            row[f"{err} 占比"] = f"{(count / total):.1%}" if total > 0 else "0.0%"
        summary_rows.append(row)

    df_tax = pd.DataFrame(summary_rows)
    df_tax.to_csv(ERROR_TAXONOMY_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return df_tax


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

    df_tax = summarize_error_taxonomy(graded_results)
    print("\n📊 Error taxonomy by category:")
    print(df_tax)

    with open(ERROR_TAXONOMY_JSON, "w", encoding="utf-8") as f:
        json.dump(clean_nan(graded_results), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
