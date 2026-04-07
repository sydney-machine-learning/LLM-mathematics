import os
import re
import json
import time
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from importlib import metadata

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT_LIBS = True
except Exception:
    HAS_PLOT_LIBS = False

from google import genai
from google.genai import types


# ======================
# Basic configuration
# ======================
API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_NAME = "gemini-2.0-flash"

DATA_PATH = "datasets/math500.jsonl"
OUTPUT_DIR = "results/gemini_math500_aligned"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "summary_report.json")
DETAILS_PATH = os.path.join(OUTPUT_DIR, "detailed_results.jsonl")
FAILED_PATH = os.path.join(OUTPUT_DIR, "failed_cases.jsonl")
CSV_PATH = os.path.join(OUTPUT_DIR, "detailed_results.csv")

MAX_RETRIES = 3
SLEEP_BETWEEN_RETRIES = 2
REQUEST_TIMEOUT = 30  # seconds

RPM = 15
K = 3  # repeat each question 3 times for consistency
ABS_TOL = 1e-6
REL_TOL = 1e-4

MOCK_MODE = False  # Set True for dry run without API

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================
# Helper classes
# ======================
class StatsCollector:
    def __init__(self):
        self.data = defaultdict(lambda: {"correct": 0, "total": 0})

    def update(self, key: str, correct: bool) -> None:
        self.data[key]["total"] += 1
        self.data[key]["correct"] += int(correct)

    def get_accuracy(self, key: str) -> float:
        total = self.data[key]["total"]
        return self.data[key]["correct"] / total if total else 0.0


class RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = rpm
        self.request_timestamps: List[float] = []

    def wait_if_needed(self) -> None:
        if self.rpm <= 0:
            return

        now = time.time()
        self.request_timestamps = [
            ts for ts in self.request_timestamps if now - ts < 60
        ]

        if len(self.request_timestamps) >= self.rpm:
            sleep_time = 60 - (now - self.request_timestamps[0]) + 0.1
            if sleep_time > 0:
                print(f"⏳ 达到 RPM={self.rpm} 限制，休眠 {sleep_time:.1f} 秒...")
                time.sleep(sleep_time)

    def mark_request(self) -> None:
        self.request_timestamps.append(time.time())


_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        if not API_KEY and not MOCK_MODE:
            raise ValueError("未检测到 GEMINI_API_KEY 环境变量")
        _client = genai.Client(
            api_key=API_KEY,
            http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT * 1000),
        )
    return _client


rate_limiter = RateLimiter(RPM)


def save_jsonl(file_path: str, rows: List[Dict[str, Any]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_answer(value: Optional[float]) -> Tuple[str, str]:
    if value is None:
        return ("N/A", "missing")
    try:
        if abs(value) >= 1e6 or (0 < abs(value) <= 1e-4):
            return (f"{value:.4e}", "scientific")
        return (f"{value:.6g}", "normal")
    except Exception as e:
        return (f"Invalid ({str(e)})", "error")


# ======================
# Numerical standardisation and extraction
# ======================
def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("＋", "+")
    text = text.replace("：", ":")
    text = text.replace("，", ",")
    text = text.strip()
    return text


def strip_boxed(text: str) -> str:
    prev = None
    current = text
    while prev != current:
        prev = current
        current = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", current)
    return current


def normalize_expression_string(text: str) -> str:
    text = normalize_text(text)
    text = strip_boxed(text)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def parse_numeric_string(value: str) -> Optional[float]:
    value = normalize_text(value)
    value = strip_boxed(value)

    try:
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            numerator, denominator = map(float, re.split(r"\s*/\s*", value))
            if denominator == 0:
                return None
            return numerator / denominator

        # Keep numeric value after removing percent sign, paper-aligned.
        if re.fullmatch(r"[-+]?\d*\.?\d+%", value):
            return float(value[:-1])

        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)

        return None
    except Exception:
        return None


NUMERIC_PATTERN = re.compile(
    r"[-+]?\d*\.?\d+%|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
)


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    text = normalize_text(text)
    text = strip_boxed(text)

    paren_match = re.fullmatch(r"\(([^()]+)\)", text)
    if paren_match:
        parsed = parse_numeric_string(paren_match.group(1).strip())
        if parsed is not None:
            return parsed

    parsed = parse_numeric_string(text)
    if parsed is not None:
        return parsed

    prefix_patterns = [
        r"^\s*answer\s*[:：]?\s*(.+?)\s*$",
        r"^\s*final answer\s*[:：]?\s*(.+?)\s*$",
        r"^\s*final result\s*[:：]?\s*(.+?)\s*$",
        r"^\s*result\s*[:：]?\s*(.+?)\s*$",
        r"^\s*the answer is\s+(.+?)\s*$",
    ]

    for pattern in prefix_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[。．.]$", "", candidate)
            candidate = strip_boxed(candidate)
            parsed = parse_numeric_string(candidate)
            if parsed is not None:
                return parsed

    # Fallback: extract the last numeric part after standardisation.
    matches = list(NUMERIC_PATTERN.finditer(text))
    if matches:
        return parse_numeric_string(matches[-1].group(0).strip())

    return None


def is_equivalent_expression(model_text: Optional[str], ref_text: Optional[str]) -> bool:
    if model_text is None or ref_text is None:
        return False

    model_norm = normalize_expression_string(model_text)
    ref_norm = normalize_expression_string(ref_text)

    if not model_norm or not ref_norm:
        return False

    if model_norm == ref_norm:
        return True

    if model_norm.startswith("(") and model_norm.endswith(")"):
        model_norm = model_norm[1:-1]
    if ref_norm.startswith("(") and ref_norm.endswith(")"):
        ref_norm = ref_norm[1:-1]

    return model_norm == ref_norm


def is_answer_correct(
    model_num: Optional[float],
    ref_num: Optional[float],
    model_text: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> bool:
    try:
        if model_num is not None and ref_num is not None:
            model = float(model_num)
            ref = float(ref_num)

            if model == ref == 0:
                return True

            abs_diff = abs(model - ref)
            rel_diff = abs_diff / (abs(ref) + 1e-9)

            if abs_diff <= ABS_TOL or rel_diff <= REL_TOL:
                return True

        if is_equivalent_expression(model_text, ref_text):
            return True

        return False

    except (TypeError, ValueError):
        return False


def majority_answer(answer_list: List[Optional[float]]) -> Optional[float]:
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return None

    counts: Dict[str, int] = defaultdict(int)
    mapping: Dict[str, float] = {}
    for ans in valid_answers:
        key = f"{ans:.12g}"
        counts[key] += 1
        mapping[key] = ans

    maj_key = max(counts, key=counts.get)
    return mapping[maj_key]


def compute_consistency(answer_list: List[Optional[float]]) -> float:
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return 0.0

    maj = majority_answer(valid_answers)
    if maj is None:
        return 0.0

    maj_key = f"{maj:.12g}"
    match_count = 0
    for ans in answer_list:
        if ans is not None and f"{ans:.12g}" == maj_key:
            match_count += 1

    return match_count / len(answer_list)


# ======================
# Gemini call with retry / timeout
# ======================
def call_gemini_api(problem: str) -> Dict[str, Any]:
    if MOCK_MODE:
        return {
            "success": True,
            "raw_output": "42",
            "error_message": None,
            "attempts_used": 1,
        }

    last_error = None
    client = get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait_if_needed()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=problem,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "Solve the mathematical problem and return only the final pure number. "
                        "Do not provide any explanation or extra text. "
                        "Do not include units, commas, or percent signs. "
                        "If the final answer is a percentage, remove the percent sign and return only the number."
                    ),
                    max_output_tokens=64,
                ),
            )
            rate_limiter.mark_request()

            raw_text = response.text.strip() if getattr(response, "text", None) else ""
            if not raw_text:
                raise RuntimeError("Empty response text from Gemini")

            return {
                "success": True,
                "raw_output": raw_text,
                "error_message": None,
                "attempts_used": attempt,
            }

        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                time.sleep(SLEEP_BETWEEN_RETRIES)
            else:
                return {
                    "success": False,
                    "raw_output": "",
                    "error_message": last_error,
                    "attempts_used": attempt,
                }

    return {
        "success": False,
        "raw_output": "",
        "error_message": last_error,
        "attempts_used": MAX_RETRIES,
    }


def check_environment() -> None:
    required_packages = ["google-genai", "pandas"]

    print("🔍 正在执行环境检查...")
    missing_packages = []

    for pkg in required_packages:
        try:
            metadata.version(pkg)
        except metadata.PackageNotFoundError:
            missing_packages.append(pkg)

    if missing_packages:
        print("❌ 环境检查失败：")
        print(f"未安装的包: {', '.join(missing_packages)}")
        print("\n💡 可执行:")
        print("pip install google-genai pandas")
        raise SystemExit(1)

    if not HAS_PLOT_LIBS:
        print("⚠️ 未检测到 matplotlib/seaborn，将跳过可视化输出。")

    print("✅ 环境检查通过")


# ======================
# Visualization
# ======================
def generate_visualizations(report: Dict[str, Any], output_dir: str = OUTPUT_DIR) -> None:
    if not HAS_PLOT_LIBS:
        print("📈 已跳过可视化图表（缺少 matplotlib/seaborn）")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    by_subject = report["stats"]["by_subject"]
    if by_subject:
        plt.figure(figsize=(12, 6))
        subjects = list(by_subject.keys())
        accuracies = [by_subject[s]["accuracy"] * 100 for s in subjects]
        sns.barplot(x=subjects, y=accuracies)
        plt.title("Subject Accuracy")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/subject_accuracy.png", dpi=300)
        plt.close()

    by_level = report["stats"]["by_level"]
    if by_level:
        plt.figure(figsize=(8, 8))
        levels = list(by_level.keys())
        counts = [by_level[lvl]["total"] for lvl in levels]
        if sum(counts) > 0:
            plt.pie(counts, labels=levels, autopct="%1.1f%%", startangle=90)
            plt.title("Difficulty Distribution")
            plt.savefig(f"{output_dir}/level_distribution.png", dpi=300)
            plt.close()

    model_types = report["stats"]["answer_types"]["model"]
    ref_types = report["stats"]["answer_types"]["reference"]
    if model_types and ref_types:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))

        sns.barplot(x=list(model_types.keys()), y=list(model_types.values()), ax=ax[0])
        ax[0].set_title("Model Answer Types")
        ax[0].set_ylabel("Count")
        ax[0].tick_params(axis="x", rotation=30)

        sns.barplot(x=list(ref_types.keys()), y=list(ref_types.values()), ax=ax[1])
        ax[1].set_title("Reference Answer Types")
        ax[1].tick_params(axis="x", rotation=30)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/answer_type_comparison.png", dpi=300)
        plt.close()

    print(f"\n📈 可视化图表已保存至 {output_dir} 目录")


# ======================
# Main evaluation
# ======================
def run_evaluation() -> Dict[str, Any]:
    check_environment()

    stats: Dict[str, Any] = {
        "global": {"correct": 0, "total": 0},
        "subjects": StatsCollector(),
        "levels": StatsCollector(),
        "answer_types": {"model": defaultdict(int), "reference": defaultdict(int)},
        "consistency_sum": 0.0,
        "consistency_count": 0,
    }

    results: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    dataset = [item for item in dataset if item.get("problem") and item.get("answer")]
    print(f"清洗后有效数据量：{len(dataset)}")

    for i, item in enumerate(dataset):
        try:
            run_outputs = []
            parsed_answers: List[Optional[float]] = []
            run_failure = False
            last_error_message = None

            for run_id in range(K):
                api_result = call_gemini_api(item["problem"])
                raw_response = api_result["raw_output"]
                parsed_answer = extract_number(raw_response)

                if not api_result["success"]:
                    run_failure = True
                    last_error_message = api_result["error_message"]

                run_outputs.append(
                    {
                        "run_id": run_id + 1,
                        "success": api_result["success"],
                        "raw_output": raw_response,
                        "parsed_answer": parsed_answer,
                        "error_message": api_result["error_message"],
                        "attempts_used": api_result["attempts_used"],
                    }
                )
                parsed_answers.append(parsed_answer)

            # Primary metric: first-pass final-answer accuracy only.
            primary_raw_response = run_outputs[0]["raw_output"]
            model_answer = run_outputs[0]["parsed_answer"]
            ref_answer = extract_number(item["answer"])

            correct = is_answer_correct(
                model_answer,
                ref_answer,
                primary_raw_response,
                item["answer"],
            )

            consistency = compute_consistency(parsed_answers)
            stats["consistency_sum"] += consistency
            stats["consistency_count"] += 1

            model_display, model_type = format_answer(model_answer)
            ref_display, ref_type = format_answer(ref_answer)

            record = {
                "Question Number": i + 1,
                "problem": item["problem"],
                "reference_text": item["answer"],
                "AI Answer": primary_raw_response,
                "Correct Answer": item["answer"],
                "Match": int(correct),
                "model_parsed": model_answer,
                "reference_parsed": ref_answer,
                "model_display": model_display,
                "reference_display": ref_display,
                "model_type": model_type,
                "reference_type": ref_type,
                "subject": item.get("subject", "Unknown"),
                "level": item.get("level", "Unknown"),
                "runs": run_outputs,
                "parsed_answers": parsed_answers,
                "consistency": consistency,
                "had_api_failure": run_failure,
            }
            results.append(record)

            stats["global"]["total"] += 1
            stats["global"]["correct"] += int(correct)
            stats["subjects"].update(record["subject"], correct)
            stats["levels"].update(record["level"], correct)
            stats["answer_types"]["model"][record["model_type"]] += 1
            stats["answer_types"]["reference"][record["reference_type"]] += 1

            if run_failure:
                failed_cases.append(
                    {
                        "Question Number": i + 1,
                        "problem": item.get("problem", ""),
                        "AI Answer": primary_raw_response or None,
                        "Correct Answer": item.get("answer", ""),
                        "Match": int(correct),
                        "subject": item.get("subject", "Unknown"),
                        "level": item.get("level", "Unknown"),
                        "error": last_error_message,
                        "is_correct": bool(correct),
                    }
                )

            print(
                f"Processed Question {i + 1}: "
                f"AI Answer = {primary_raw_response}, "
                f"Correct = {item['answer']}, Match = {int(correct)}"
            )

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            print(f"🔥 Q{i + 1} {error_msg}")

            failed_record = {
                "Question Number": i + 1,
                "problem": item.get("problem", ""),
                "AI Answer": None,
                "Correct Answer": item.get("answer", ""),
                "Match": 0,
                "subject": item.get("subject", "Unknown"),
                "level": item.get("level", "Unknown"),
                "error": error_msg,
                "is_correct": False,
            }
            failed_cases.append(failed_record)
            results.append(failed_record)

            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "Unknown"), False)
            stats["levels"].update(item.get("level", "Unknown"), False)

    global_accuracy = (
        stats["global"]["correct"] / stats["global"]["total"]
        if stats["global"]["total"]
        else 0.0
    )
    average_consistency = (
        stats["consistency_sum"] / stats["consistency_count"]
        if stats["consistency_count"]
        else 0.0
    )

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(dataset),
            "config": {
                "model": MODEL_NAME,
                "timeout": REQUEST_TIMEOUT,
                "max_retries": MAX_RETRIES,
                "abs_tol": ABS_TOL,
                "rel_tol": REL_TOL,
                "K": K,
                "primary_metric": "first_pass_final_answer_accuracy",
                "consistency_metric": "supplementary_multi_run_consistency",
                "prompt_mode": "numeric_only",
                "mock_mode": MOCK_MODE,
            },
        },
        "stats": {
            "global_accuracy": global_accuracy,
            "global_total": stats["global"]["total"],
            "global_correct": stats["global"]["correct"],
            "average_consistency": average_consistency,
            "by_subject": {
                s: {
                    "accuracy": stats["subjects"].get_accuracy(s),
                    "correct": stats["subjects"].data[s]["correct"],
                    "total": stats["subjects"].data[s]["total"],
                }
                for s in stats["subjects"].data
            },
            "by_level": {
                l: {
                    "accuracy": stats["levels"].get_accuracy(l),
                    "correct": stats["levels"].data[l]["correct"],
                    "total": stats["levels"].data[l]["total"],
                }
                for l in stats["levels"].data
            },
            "answer_types": {
                "model": dict(stats["answer_types"]["model"]),
                "reference": dict(stats["answer_types"]["reference"]),
            },
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    save_jsonl(DETAILS_PATH, results)
    save_jsonl(FAILED_PATH, failed_cases)
    pd.DataFrame(results).to_csv(CSV_PATH, index=False)

    generate_visualizations(report, OUTPUT_DIR)

    print("\n📊 评估报告")
    print(f"总题数: {report['metadata']['data_size']}")
    print(f"总准确率: {report['stats']['global_accuracy']:.2%}")
    print(f"平均一致性 (K={K}): {report['stats']['average_consistency']:.4f}")
    print(f"失败样本数: {len(failed_cases)}")

    return report


# ======================
# Test extractor
# ======================
def test_extractor() -> None:
    test_cases = [
        ("42", 42.0),
        ("3.1416", 3.1416),
        ("1.23e5", 123000.0),
        ("11/2", 5.5),
        ("\\boxed{11/2}", 5.5),
        ("5%", 5.0),
        ("answer: 42", 42.0),
        ("final answer: \\boxed{3.14}", 3.14),
        ("The answer is 42.", 42.0),
        ("最后数值是42", 42.0),
        ("1,200%", 1200.0),
        ("1200.0", 1200.0),
        ("无效答案", None),
    ]

    print("\n🔬 正在执行解析测试...")

    for text, expected in test_cases:
        result = extract_number(text)
        if result is None or expected is None:
            success = result == expected
        else:
            success = abs(result - expected) < 1e-6

        display, _ = format_answer(result)
        exp_display, _ = format_answer(expected)
        status = "✅" if success else "❌"
        print(f"{status} {text} → {display} (预期: {exp_display})")

    print("🎉 所有解析测试完成")


if __name__ == "__main__":
    test_extractor()
    run_evaluation()
