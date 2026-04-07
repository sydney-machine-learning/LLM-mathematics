import json
import re
import sys
import time
import os
import requests
from tqdm import tqdm
from collections import defaultdict
import importlib.metadata as metadata
from typing import Optional, Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


# ======================
# Basic setting
# ======================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

DATA_PATH = "datasets/math500.jsonl"
OUTPUT_DIR = "results/deepseek_math500"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "deepseek_math500_report.json")
DETAILS_PATH = os.path.join(OUTPUT_DIR, "deepseek_math500_results.jsonl")
FAILED_PATH = os.path.join(OUTPUT_DIR, "deepseek_math500_failed_cases.jsonl")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 45

ABS_TOL = 1e-4
REL_TOL = 1e-4

# Supplementary consistency only
K = 3

# If you cannot get the API key, you can open MOCK_MODE to run through the process first
MOCK_MODE = False
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================
# Statistical module
# ======================
class StatsCollector:
    def __init__(self):
        self.data = defaultdict(lambda: {"correct": 0, "total": 0})

    def update(self, key: str, is_correct: bool) -> None:
        self.data[key]["total"] += 1
        if is_correct:
            self.data[key]["correct"] += 1

    def get_accuracy(self, key: str) -> float:
        return self.data[key]["correct"] / self.data[key]["total"] if self.data[key]["total"] else 0.0


def retry_decorator():
    from tenacity import retry, stop_after_attempt, wait_exponential
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )


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
def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("−", "-")
    text = text.replace("＋", "+")
    text = text.replace("：", ":")
    text = text.strip()
    return text


def normalize_expression_string(text: str) -> str:
    """
    Lightweight normalization for equivalent-expression comparison.
    This is intentionally simple to stay close to the paper description
    without introducing a full symbolic prover.
    """
    text = normalize_text(text)
    text = text.replace("\\boxed{", "").replace("}", "")
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".")
    return text


def parse_numeric_string(value: str) -> Optional[float]:
    value = normalize_text(value)

    try:
        # fraction
        if re.fullmatch(r"[-+]?\d+\s*/\s*[-+]?\d+", value):
            numerator, denominator = map(float, re.split(r"\s*/\s*", value))
            if denominator == 0:
                return None
            return numerator / denominator

        # percentage -> decimal
        if re.fullmatch(r"[-+]?\d*\.?\d+%", value):
            return float(value[:-1]) / 100.0

        # plain number / scientific notation
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)

        return None
    except Exception:
        return None


def extract_number(text: str) -> Optional[float]:
    """
    Paper-aligned extraction:
    1) standardise formatting
    2) handle boxed / fraction / percentage / plain number
    3) allow limited prefixes like 'answer:'
    4) fallback to regex extraction of the last numeric token,
       consistent with the paper's description of extracting
       the last digit/numeric part using regular expressions.
    """
    if not text:
        return None

    text = normalize_text(text)

    # \boxed{...}
    boxed_match = re.fullmatch(r"\\boxed\{([^{}]+)\}", text)
    if boxed_match:
        parsed = parse_numeric_string(boxed_match.group(1).strip())
        if parsed is not None:
            return parsed

    # (...) wrapper
    paren_match = re.fullmatch(r"\(([^()]+)\)", text)
    if paren_match:
        parsed = parse_numeric_string(paren_match.group(1).strip())
        if parsed is not None:
            return parsed

    # direct numeric answer
    parsed = parse_numeric_string(text)
    if parsed is not None:
        return parsed

    # limited prefixes
    prefix_patterns = [
        r"^\s*answer\s*[:：]?\s*(.+?)\s*$",
        r"^\s*final answer\s*[:：]?\s*(.+?)\s*$",
        r"^\s*final result\s*[:：]?\s*(.+?)\s*$",
        r"^\s*result\s*[:：]?\s*(.+?)\s*$",
    ]

    for pattern in prefix_patterns:
        match = re.fullmatch(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            candidate = re.sub(r"[。．.]$", "", candidate)

            boxed_match = re.fullmatch(r"\\boxed\{([^{}]+)\}", candidate)
            if boxed_match:
                candidate = boxed_match.group(1).strip()

            parsed = parse_numeric_string(candidate)
            if parsed is not None:
                return parsed

    # fallback: extract the final numeric token after standardisation
    # match full tokens first so fractions/percentages are not broken into plain numbers
    combined_pattern = re.compile(
        r"[-+]?\d*\.?\d+%|[-+]?\d+\s*/\s*[-+]?\d+|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    )
    matches = list(combined_pattern.finditer(text))

    if matches:
        value = matches[-1].group(0).strip()

        try:
            if value.endswith("%"):
                return float(value[:-1]) / 100.0
            if "/" in value:
                numerator, denominator = map(float, re.split(r"\s*/\s*", value))
                if denominator == 0:
                    return None
                return numerator / denominator
            return float(value)
        except Exception:
            return None

    return None


def is_equivalent_expression(model_text: Optional[str], ref_text: Optional[str]) -> bool:
    """
    Lightweight equivalent-expression support, consistent with the paper's
    statement that equivalent expressions may be accepted.
    This is intentionally conservative.
    """
    if model_text is None or ref_text is None:
        return False

    model_norm = normalize_expression_string(model_text)
    ref_norm = normalize_expression_string(ref_text)

    if not model_norm or not ref_norm:
        return False

    # exact normalized string match
    if model_norm == ref_norm:
        return True

    # simple outer parentheses normalization
    if model_norm.startswith("(") and model_norm.endswith(")"):
        model_norm = model_norm[1:-1]
    if ref_norm.startswith("(") and ref_norm.endswith(")"):
        ref_norm = ref_norm[1:-1]

    return model_norm == ref_norm


def is_answer_correct(
    model_num: Optional[float],
    ref_num: Optional[float],
    model_text: Optional[str] = None,
    ref_text: Optional[str] = None
) -> bool:
    """
    Main paper-aligned judgment:
    1) numeric comparison with tolerance
    2) lightweight equivalent-expression comparison
    """
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

        # simple equivalent-expression acceptance
        if is_equivalent_expression(model_text, ref_text):
            return True

        return False

    except (TypeError, ValueError):
        return False


def majority_answer(answer_list: List[Optional[float]]) -> Optional[float]:
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return None

    counts = defaultdict(int)
    mapping = {}
    for ans in valid_answers:
        key = f"{ans:.12g}"
        counts[key] += 1
        mapping[key] = ans

    maj_key = max(counts, key=counts.get)
    return mapping[maj_key]


def compute_consistency(answer_list: List[Optional[float]]) -> float:
    """
    Supplementary only: multi-run agreement across K runs.
    """
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
# API call
# ======================
@retry_decorator()
def call_deepseek_api(problem: str) -> Dict[str, Any]:
    if MOCK_MODE:
        fake_output = "42"
        return {
            "success": True,
            "raw_output": fake_output,
            "error_message": None,
            "attempts_used": 1
        }

    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "MathEvaluator/FinalRepo"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a math expert. Return only the final numeric answer. "
                    "Do not provide any explanation, units, commas, percent signs, or extra text. "
                    "If the answer is a percentage, convert it to decimal form."
                )
            },
            {
                "role": "user",
                "content": problem
            }
        ],
        "max_tokens": 64,
        "temperature": 0
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            error_info = {
                "status": response.status_code,
                "headers": dict(response.headers),
                "body": response.text[:500]
            }
            raise requests.HTTPError(json.dumps(error_info, ensure_ascii=False, indent=2))

        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()

        return {
            "success": True,
            "raw_output": content,
            "error_message": None,
            "attempts_used": 1
        }

    except requests.exceptions.Timeout as e:
        raise RuntimeError(f"请求超时: {str(e)}")
    except requests.exceptions.SSLError as e:
        raise RuntimeError(f"SSL证书错误: {str(e)}")
    except requests.exceptions.ProxyError as e:
        raise RuntimeError(f"代理错误: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"请求失败: {str(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"响应解析失败: {str(e)}\noriginal response: {response.text[:500]}")
    except KeyError as e:
        raise RuntimeError(f"API响应字段缺失: {str(e)}\nresponse: {response.text[:500]}")
    except Exception as e:
        raise RuntimeError(str(e))


def check_environment():
    required_packages = [
        "tenacity",
        "requests",
        "tqdm",
        "matplotlib",
        "seaborn"
    ]

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
        print("pip install tenacity requests tqdm matplotlib seaborn")
        sys.exit(1)

    print("✅ 环境检查通过")


# ======================
# 可视化模块
# ======================
def generate_visualizations(report: dict, output_dir: str = OUTPUT_DIR):
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
# 主评估流程
# ======================
def run_evaluation():
    check_environment()

    stats = {
        "global": {"correct": 0, "total": 0},
        "subjects": StatsCollector(),
        "levels": StatsCollector(),
        "answer_types": {"model": defaultdict(int), "reference": defaultdict(int)},
        "consistency_sum": 0.0,
        "consistency_count": 0
    }

    results = []
    failed_cases = []

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    dataset = [item for item in dataset if item.get("problem") and item.get("answer")]
    print(f"清洗后有效数据量：{len(dataset)}")

    progress_bar = tqdm(
        dataset,
        desc="🔧 评估进度",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [准确率: {postfix}]"
    )

    for item in progress_bar:
        try:
            run_outputs = []
            parsed_answers = []

            for run_id in range(K):
                api_result = call_deepseek_api(item["problem"])
                raw_response = api_result["raw_output"]
                parsed_answer = extract_number(raw_response)

                run_outputs.append({
                    "run_id": run_id + 1,
                    "success": api_result["success"],
                    "raw_output": raw_response,
                    "parsed_answer": parsed_answer,
                    "error_message": api_result["error_message"],
                    "attempts_used": api_result["attempts_used"]
                })
                parsed_answers.append(parsed_answer)

            # Main metric: first-pass final-answer accuracy only
            primary_raw_response = run_outputs[0]["raw_output"]
            model_answer = run_outputs[0]["parsed_answer"]
            ref_answer = extract_number(item["answer"])

            correct = is_answer_correct(
                model_answer,
                ref_answer,
                primary_raw_response,
                item["answer"]
            )

            # supplementary consistency only
            consistency = compute_consistency(parsed_answers)
            stats["consistency_sum"] += consistency
            stats["consistency_count"] += 1

            model_display, model_type = format_answer(model_answer)
            ref_display, ref_type = format_answer(ref_answer)

            record = {
                "problem": item["problem"],
                "reference_text": item["answer"],
                "model_raw": model_answer,
                "reference_raw": ref_answer,
                "model_display": model_display,
                "reference_display": ref_display,
                "model_type": model_type,
                "reference_type": ref_type,
                "is_correct": correct,
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "raw_response": primary_raw_response,
                "runs": run_outputs,
                "parsed_answers": parsed_answers,
                "consistency": consistency
            }
            results.append(record)

            stats["global"]["total"] += 1
            stats["global"]["correct"] += int(correct)
            stats["subjects"].update(record["subject"], correct)
            stats["levels"].update(record["level"], correct)
            stats["answer_types"]["model"][record["model_type"]] += 1
            stats["answer_types"]["reference"][record["reference_type"]] += 1

            current_acc = (
                stats["global"]["correct"] / stats["global"]["total"]
                if stats["global"]["total"] else 0.0
            )
            progress_bar.set_postfix_str(f"{current_acc:.1%}")

            tqdm.write(
                f"{'✅' if correct else '❌'} {item['problem'][:40]}... | "
                f"参考: {ref_display} vs 模型: {model_display}"
            )

        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            tqdm.write(f"🔥 {error_msg}")

            failed_record = {
                "problem": item.get("problem", ""),
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "error": error_msg,
                "is_correct": False
            }
            failed_cases.append(failed_record)
            results.append(failed_record)

            # Failed or unparsable cases are counted as incorrect
            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "unknown"), False)
            stats["levels"].update(item.get("level", "unknown"), False)

            current_acc = (
                stats["global"]["correct"] / stats["global"]["total"]
                if stats["global"]["total"] else 0.0
            )
            progress_bar.set_postfix_str(f"{current_acc:.1%}")

    global_accuracy = (
        stats["global"]["correct"] / stats["global"]["total"]
        if stats["global"]["total"] else 0.0
    )
    average_consistency = (
        stats["consistency_sum"] / stats["consistency_count"]
        if stats["consistency_count"] else 0.0
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
                "mock_mode": MOCK_MODE
            }
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
                    "total": stats["subjects"].data[s]["total"]
                }
                for s in stats["subjects"].data
            },
            "by_level": {
                l: {
                    "accuracy": stats["levels"].get_accuracy(l),
                    "correct": stats["levels"].data[l]["correct"],
                    "total": stats["levels"].data[l]["total"]
                }
                for l in stats["levels"].data
            },
            "answer_types": {
                "model": dict(stats["answer_types"]["model"]),
                "reference": dict(stats["answer_types"]["reference"])
            }
        }
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    save_jsonl(DETAILS_PATH, results)
    save_jsonl(FAILED_PATH, failed_cases)

    generate_visualizations(report, OUTPUT_DIR)

    print("\n📊 评估报告")
    print(f"总题数: {report['metadata']['data_size']}")
    print(f"总准确率: {report['stats']['global_accuracy']:.2%}")
    print(f"平均一致性 (K={K}): {report['stats']['average_consistency']:.4f}")
    print(f"失败样本数: {len(failed_cases)}")

    return report


# ======================
# 测试模块
# ======================
def test_extractor():
    test_cases = [
        ("42", 42.0),
        ("3.1416", 3.1416),
        ("1.23e5", 123000.0),
        ("11/2", 5.5),
        ("\\boxed{11/2}", 5.5),
        ("5%", 0.05),
        ("answer: 42", 42.0),
        ("final answer: \\boxed{3.14}", 3.14),
        ("The answer is 42.", 42.0),
        ("最后数值是42", 42.0),
        ("1,200%", 12.0),
        ("1200.0", 1200.0),
        ("无效答案", None)
    ]

    print("\n🔬 正在执行解析测试...")
    try:
        print(f"测试环境版本: tenacity {metadata.version('tenacity')}")
    except Exception:
        pass

    for text, expected in test_cases:
        result = extract_number(text)
        if result is None or expected is None:
            success = (result == expected)
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
