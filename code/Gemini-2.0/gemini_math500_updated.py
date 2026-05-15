import json
import re
import sys
import time
import os
import logging
import requests
import importlib.metadata as metadata
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Dict, Any, Tuple, List
from requests.exceptions import Timeout, SSLError, ProxyError, RequestException, ConnectionError
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT_LIBS = True
except Exception:
    HAS_PLOT_LIBS = False

from google import genai
from google.genai import types



# Basic configuration
API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set API key before running
MODEL_NAME = "gemini-2.5-flash"


DATA_PATH = os.getenv("MATH500_DATA_PATH", "datasets/math500.jsonl")

OUTPUT_DIR = "results/gemini_2.5_math500"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gemini_2.5_math500_report.json")
DETAILS_PATH = os.path.join(OUTPUT_DIR, "gemini_2.5_math500_results.jsonl")
FAILED_PATH = os.path.join(OUTPUT_DIR, "gemini_2.5_math500_failed_cases.jsonl")

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60

ABS_TOL = 1e-4
REL_TOL = 1e-4

# Consistency
K = 3
MOCK_MODE = False

os.makedirs(OUTPUT_DIR, exist_ok=True)



# Logging 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'evaluation.log'), encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)



# Optional symbolic comparison support

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    HAS_SYMPY = True
    logger.info("Sympy is available - symbolic expression comparison enabled")
except ImportError:
    HAS_SYMPY = False
    logger.warning("Sympy is not available - symbolic expression comparison disabled")


# Statistics helper

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
    """Use tenacity if available; otherwise fall back to a small built-in retry wrapper."""
    try:
        from tenacity import retry, stop_after_attempt, wait_exponential
        return retry(
            stop=stop_after_attempt(MAX_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
    except ModuleNotFoundError:
        def decorator(func):
            def wrapper(*args, **kwargs):
                last_error = None
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {str(e)}")
                        if attempt < MAX_RETRIES:
                            time.sleep(min(2 ** attempt, 10))
                raise last_error
            return wrapper
        return decorator


def save_jsonl(file_path: str, rows: List[Dict[str, Any]]) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# Answer cleaning and extraction

UNICODE_REPLACEMENTS = {
    "π": r"\pi",
    "∞": r"\infty",
    "∪": r"\cup",
    "∩": r"\cap",
    "−": "-",
    "–": "-",
    "—": "-",
    "＋": "+",
    "≥": r"\ge",
    "≤": r"\le",
    "∈": r"\in",
    "×": "*",
    "÷": "/",
}


def normalize_unicode_math(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text).strip()

    # Unicode subscript digits: 2516₈ -> 2516_8
    subscript_map = {
        "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
        "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
    }

    def replace_subscript(match):
        digits = "".join(subscript_map[ch] for ch in match.group(0))
        return "_" + digits

    text = re.sub(r"[₀₁₂₃₄₅₆₇₈₉]+", replace_subscript, text)

    # Unicode sqrt: √13 -> \sqrt{13}, 3√13 -> 3\sqrt{13}
    text = re.sub(r"√\s*\{([^{}]+)\}", r"\\sqrt{\1}", text)
    text = re.sub(r"√\s*([A-Za-z0-9]+)", r"\\sqrt{\1}", text)

    for k, v in UNICODE_REPLACEMENTS.items():
        text = text.replace(k, v)

    return text


def strip_math_delimiters(text: str) -> str:
    text = text.strip()
    # Remove common whole-answer math wrappers.
    changed = True
    while changed:
        changed = False
        wrappers = [
            (r"^\$\$(.*)\$\$$", 1),
            (r"^\$(.*)\$$", 1),
            (r"^\\\((.*)\\\)$", 1),
            (r"^\\\[(.*)\\\]$", 1),
        ]
        for pattern, group_idx in wrappers:
            m = re.fullmatch(pattern, text, flags=re.DOTALL)
            if m:
                text = m.group(group_idx).strip()
                changed = True
    return text


def find_latex_command_contents(text: str, command: str) -> List[str]:
    r"""Return all top-level contents of \command{...}, supporting nested braces."""
    contents = []
    pattern = "\\\\" + re.escape(command) + r"\s*\{"
    for m in re.finditer(pattern, text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            contents.append(text[start:i - 1])
    return contents


def strip_latex_wrappers(text: Optional[str]) -> str:
    if text is None:
        return ""
    
    text = normalize_unicode_math(text)
    text = strip_math_delimiters(text)
    text = text.replace(r"\left", "").replace(r"\right", "").strip()
    
    wrapper_cmds = ["boxed", "fbox", "text", "mathrm", "mbox", "mathbf", "mathit"]
    
    while True:
        found = False
        for cmd in wrapper_cmds:
            cmd_pattern = "\\" + cmd
            if text.startswith(cmd_pattern):
                start_idx = len(cmd_pattern)
                rest = text[start_idx:].strip()
                if rest.startswith("{"):
                    depth = 0
                    for i, ch in enumerate(rest):
                        if ch == "{":
                            depth += 1
                        elif ch == "}":
                            depth -= 1
                            if depth == 0:
                                inner = rest[1:i].strip()
                                text = strip_math_delimiters(inner)
                                found = True
                                break
                if found:
                    break
        if not found:
            break
    
    # Remove the outer parentheses carefully.
    while (text.startswith("(") and text.endswith(")") and 
           text.count("(") == text.count(")") == 1):
        inner = text[1:-1].strip()
         # If the content contains commas, equal signs, and inequality symbols internally, the parentheses shall be retained.
        if any(c in inner for c in [',', '=', '<', '>', r'\le', r'\ge', r'\in']):
            break
        # If there are no operators inside, parentheses are likely required.
        if not any(c in inner for c in ['+', '-', '*', '/', '^']):
            break
        text = inner
    
    return text.strip()


def extract_final_answer_text(text: Optional[str]) -> str:
    if not text:
        return ""
    text_str = str(text).strip()
    
    # 1.Prioritise the extraction of boxed content (nesting is supported).
    boxed_contents = find_latex_command_contents(text_str, "boxed")
    if boxed_contents:
        # Recursively extract nested boxed elements
        result = boxed_contents[-1].strip()
        inner_boxed = find_latex_command_contents(result, "boxed")
        if inner_boxed:
            return inner_boxed[-1].strip()
        return result
    
    # 2.Attempt to identify the content following conclusive conjunctions such as "therefore" and "thus"
    conclusion_patterns = [
        r"(?i)(?:therefore|thus|hence|so|finally|conclusion)\s*[,:]*\s*(.+)$",
        r"(?i)(?:答案|因此|所以|故|综上|最终)\s*[：:，,]*\s*(.+)$",
    ]
    for pattern in conclusion_patterns:
        matches = re.findall(pattern, text_str, re.MULTILINE)
        if matches:
            # Take the last conclusion.
            candidate = matches[-1].strip()
            # If the conclusion is a valid mathematical expression, return it directly.
            if any(c in candidate for c in '=+\-*/^0123456789') or '\\' in candidate:
                return candidate
    
    # 3.The Original Extraction Logic
    patterns = [
        r"(?i)(?:final\s+answer|answer|result)\s*[:：=]\s*(.+)",
        r"(?i)the\s+answer\s+is\s+(.+)",
        r"(?i)(?:is|equals|equal to)\s*\\?\(?\$?\s*([-+]?\d+(?:\.\d+)?)\s*\\?\)?\$?\.?\s*$",
    ]
    for p in patterns:
        match = re.search(p, text_str, flags=re.DOTALL)
        if match:
            candidate = match.group(1).strip()
            # Attempt to perform truncation at the first period or line break.
            candidate = re.split(r'[。．.]\s|\n', candidate)[0]
            return candidate.strip()
    
    # 4.Check for the existence of LaTeX mathematical environments
    math_envs = re.findall(r'\$(.+?)\$|\\\[(.+?)\\\]', text_str, re.DOTALL)
    if math_envs:
        # Take the last mathematical expression
        last_math = math_envs[-1]
        return (last_math[0] or last_math[1]).strip()
    
    # 5. Final fallback: Retrieve the last line that appears to be the answer.
    lines = text_str.split("\n")
    # Find the first line that appears to be a mathematical answer when traversing from the end to the beginning.
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        # Omit lines that are clearly inference steps.
        if re.match(r'^(Let|Suppose|Assume|We|First|Step|Note|Compute|Solve)\b', line):
            continue
        # If mathematical symbols are included, it is most likely the answer.
        if any(c in line for c in '=+\-*/^') or re.search(r'\\[a-zA-Z]+', line):
            return line
        # If numeric values are present, they may constitute the solution.
        if re.search(r'\d', line):
            return line
    
    # 6. Final fallback
    if lines:
        return lines[-1].strip()
    
    return text_str


def extract_rhs_if_equation_chain(text: Optional[str]) -> str:
    if not text:
        return ""

    raw = extract_final_answer_text(text)
    raw = strip_latex_wrappers(raw)
    raw = strip_math_delimiters(raw)
    raw = raw.strip()

    # Do not touch inequalities.
    if any(sym in raw for sym in ["<", ">", r"\le", r"\ge", "≤", "≥"]):
        return raw

    if "=" not in raw:
        return raw

    parts = [p.strip() for p in raw.split("=") if p.strip()]
    if len(parts) < 2:
        return raw

    lhs = parts[0]
    rhs = parts[-1]

    # Remove LaTeX command names before checking variables.
    lhs_for_check = normalize_unicode_math(lhs)
    lhs_for_check = re.sub(r"\\[a-zA-Z]+", "", lhs_for_check)
    lhs_for_check = re.sub(r"\{[^{}]*\}", "", lhs_for_check)

    # If the left-hand side contains real algebraic variables, this is probably
    # a final equation such as y=2x+3 or 5x-7y+11z+4=0. Do NOT reduce it to RHS.
    # Treat i/I as imaginary unit, not as an algebraic variable.
    variable_letters = set(re.findall(r"[A-Za-z]", lhs_for_check))
    variable_letters = {ch.lower() for ch in variable_letters if ch.lower() != "i"}

    if variable_letters:
        return raw

    # Safe case:
    # \frac{5}{5} + \frac{10}{5}i = 1 + 2i
    # 2+2=4
    return rhs

def extract_rhs_from_single_variable_equation(text: Optional[str]) -> Optional[str]:
    """
    Extract RHS from simple final-answer equations like:
    r = \sqrt{5}
    x = 5
    k = -2

    Do not extract from full equations such as:
    y = 2x + 3
    5x - 7y + 11z + 4 = 0
    """
    if not text:
        return None

    raw = extract_final_answer_text(text)
    raw = strip_latex_wrappers(raw)
    raw = strip_math_delimiters(raw).strip()

    if "=" not in raw:
        return None

    parts = [p.strip() for p in raw.split("=") if p.strip()]
    if len(parts) != 2:
        return None

    lhs, rhs = parts

    # Only allow a single variable on the left.
    if not re.fullmatch(r"[A-Za-z]", lhs):
        return None

    # If RHS still contains the same variable, it is likely a real equation, not a value assignment.
    # Example: y = 2x + 3 should not become 2x+3 blindly.
    if re.search(rf"\b{re.escape(lhs)}\b", rhs):
        return None

    return rhs

def strip_numeric_units(text: Optional[str]) -> str:
    """Remove units and degree symbols without damaging LaTeX fractions or expressions."""
    if not text:
        return ""

    text = normalize_unicode_math(text)
    text = strip_math_delimiters(text)
    text = text.replace(r"\left", "").replace(r"\right", "")

    # Remove LaTeX text unit boxes, including exponents: \mbox{ cm}^2, \text{m}^{3}
    text = re.sub(r"\\mbox\{\s*[A-Za-z]+\s*\}(?:\^\{?\d+\}?)?", "", text)
    text = re.sub(r"\\text\{\s*[A-Za-z]+\s*\}(?:\^\{?\d+\}?)?", "", text)

    # Remove spacing commands often placed before units or currency.
    text = re.sub(r"\\[,;! ]", "", text)
    text = text.replace("~", " ")

    # Remove currency / malformed math delimiters around numeric answers.
    # Examples: \$32,\!348 -> 32348, $78 -> 78
    text = text.replace(r"\$", "")
    text = text.replace("$", "")
    # Remove degree symbols: ^\circ, ^{\circ}, \degree, °
    text = re.sub(r"\s*\^\s*\{?\s*\\circ\s*\}?", "", text)
    text = re.sub(r"\s*\\circ\b", "", text)
    text = re.sub(r"\s*\\degree\b", "", text)
    text = re.sub(r"\s*°", "", text)

    # Remove common units after a number, with optional squared/cubed exponent.
    units = [
        "centimeters", "centimetres", "centimeter", "centimetre", "inches", "inch",
        "meters", "metres", "meter", "metre", "kilometers", "kilometres", "kilometer", "kilometre",
        "millimeters", "millimetres", "millimeter", "millimetre", "feet", "foot", "yards", "yard",
        "miles", "mile", "grams", "gram", "kilograms", "kilogram", "milligrams", "milligram",
        "pounds", "pound", "ounces", "ounce", "units", "unit", "cm", "mm", "km", "kg", "mg", "ml",
        "ft", "yd", "mi", "lb", "oz", "in", "m", "g", "l",
    ]
    units_pattern = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
    text = re.sub(
        rf"(?<=\d)\s*(?:square|sq\.?|cubic)?\s*(?:{units_pattern})\b\s*(?:\^\s*\{{?\d+\}}?)?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove trailing unit words such as "square units" after the previous pass.
    text = re.sub(r"(?<=\d)\s*(?:square|sq\.?|cubic)\s+units?\b", "", text, flags=re.IGNORECASE)

    return text.strip()


def format_numeric_answer(value: Optional[float]) -> Tuple[str, str]:
    if value is None:
        return ("N/A", "missing")
    try:
        if abs(value) >= 1e6 or (0 < abs(value) <= 1e-4):
            return (f"{value:.4e}", "numeric")
        return (f"{value:.6g}", "numeric")
    except Exception as e:
        return (f"Invalid ({str(e)})", "error")


def parse_numeric_string(value: Optional[str]) -> Optional[float]:
    if not value:
        return None

    value = extract_final_answer_text(value)
    value = strip_latex_wrappers(value)
    value = strip_numeric_units(value)
    value = normalize_unicode_math(value)
    value = value.replace(",", "").replace("，", "")
    value = value.strip()
    value = re.sub(r"\s+", "", value)
    
    # Processing of percentage signs
    value = value.replace(r"\%", "%")
    value = value.replace("\\%", "%")
    
    # Percentage Matching
    if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)%", value):
        return float(value[:-1]) / 100.0
    
    if looks_like_base_notation(value):
        return None

    # Remove common trailing punctuation.
    value = re.sub(r"[。．.]$", "", value)

    try:
        # Signed LaTeX fraction: -\frac{24}{25}
        signed_frac_match = re.fullmatch(
            r"([-+]?)\\(?:dfrac|tfrac|frac)\s*\{\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}\s*\{\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}",
            value,
        )
        if signed_frac_match:
            sign = -1.0 if signed_frac_match.group(1) == "-" else 1.0
            numerator = float(signed_frac_match.group(2))
            denominator = float(signed_frac_match.group(3))
            return sign * numerator / denominator if denominator != 0 else None
        
        # Mixed number: 137 \frac{1}{2}
        mixed_latex_match = re.fullmatch(
            r"([-+]?\d+)\s*\\frac\{(\d+)\}\{(\d+)\}",
            value
        )
        if mixed_latex_match:
            whole = float(mixed_latex_match.group(1))
            numerator = float(mixed_latex_match.group(2))
            denominator = float(mixed_latex_match.group(3))
            if denominator == 0:
                return None
            sign = -1 if whole < 0 else 1
            return whole + sign * (numerator / denominator)

        # Mixed number: 137 1/2
        mixed_plain_match = re.fullmatch(
            r"([-+]?\d+)\s+(\d+)\s*/\s*(\d+)",
            value
        )
        if mixed_plain_match:
            whole = float(mixed_plain_match.group(1))
            numerator = float(mixed_plain_match.group(2))
            denominator = float(mixed_plain_match.group(3))
            if denominator == 0:
                return None
            sign = -1 if whole < 0 else 1
            return whole + sign * (numerator / denominator)

        # LaTeX fraction: \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b}
        frac_match = re.fullmatch(
            r"\\(?:dfrac|tfrac|frac)\s*\{\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}\s*\{\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}",
            value,
        )
        if frac_match:
            numerator = float(frac_match.group(1))
            denominator = float(frac_match.group(2))
            return numerator / denominator if denominator != 0 else None

        # Loose LaTeX fraction after backslash/braces were omitted: frac{a}{b}, frac a b, fracab.
        loose_frac_match = re.fullmatch(
            r"\\?(?:dfrac|tfrac|frac)\s*\{?\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}?\s*\{?\s*([-+]?(?:\d+\.?\d*|\.\d+))\s*\}?",
            value,
        )
        if loose_frac_match:
            numerator = float(loose_frac_match.group(1))
            denominator = float(loose_frac_match.group(2))
            return numerator / denominator if denominator != 0 else None

        # Plain fraction: a/b.
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)\s*/\s*[-+]?(?:\d+\.?\d*|\.\d+)", value):
            numerator, denominator = map(float, re.split(r"\s*/\s*", value))
            return numerator / denominator if denominator != 0 else None

        # Plain number / scientific notation.
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", value):
            return float(value)

        return None
    except Exception as e:
        logger.debug(f"Failed to parse numeric string '{value}': {str(e)}")
        return None


def extract_number_from_simple_equation(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text)
    text = strip_numeric_units(text)
    text = normalize_unicode_math(text)
    text = strip_math_delimiters(text)
    text = text.replace(" ", "")

    # Do not treat base notation like 4210_5 as numeric
    if looks_like_base_notation(text):
        return None

    # Examples: x=5, r=4, k = -2
    m = re.fullmatch(r"[a-zA-Z]+\s*=\s*(.+)", text)
    if m:
        return parse_numeric_string(m.group(1))

    return None


def extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None

    eq_num = extract_number_from_simple_equation(text)
    if eq_num is not None:
        return eq_num

    candidate = extract_final_answer_text(text)
    parsed = parse_numeric_string(candidate)
    if parsed is not None:
        return parsed

    cleaned = strip_numeric_units(candidate)
    cleaned = cleaned.replace(",", "").replace("，", "")

    cleaned_norm = normalize_unicode_math(cleaned)

    # Do not extract partial numbers from symbolic math expressions.
    # Example: 18+2\pi should not be parsed as 2.
    # But allow prose like: "The answer is 34."
    if re.search(r"\\(?:pi|sqrt|sin|cos|tan|cot|sec|csc|log|ln)|[A-Za-z]", cleaned_norm):
        if re.search(r"(?i)\b(?:answer|result|equals|equal to|is)\b", cleaned_norm):
            pass
        else:
            return None

    # Fallback: take the last numeric-looking token.
    combined_pattern = re.compile(
        r"[-+]?(?:\d+\.?\d*|\.\d+)%|[-+]?(?:\d+\.?\d*|\.\d+)\s*/\s*[-+]?(?:\d+\.?\d*|\.\d+)|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
    )
    matches = list(combined_pattern.finditer(cleaned))
    if not matches:
        return None

    value = matches[-1].group(0).strip()
    return parse_numeric_string(value)


def normalize_text(text: Optional[str]) -> str:
    """Aggressive normalisation for simple text/string comparison only."""
    if not text:
        return ""
    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text)
    text = strip_numeric_units(text)
    text = normalize_unicode_math(text)
    text = text.replace(r"\left", "").replace(r"\right", "")
    text = re.sub(r"\\(?:text|mathrm|mbox|mathbf|mathit)\s*\{([^{}]+)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    return text.lower().strip()


def strip_redundant_outer_parens(text: str) -> str:
    text = text.strip()
    while True:
        m = re.fullmatch(r"\(([A-Za-z0-9\\+\-]+)\)", text)
        if not m:
            break
        text = m.group(1).strip()
    return text


def normalize_text_answer(text: Optional[str]) -> str:
    text = normalize_text(text)
    return strip_redundant_outer_parens(text)


def normalize_expression_string(text: Optional[str]) -> str:
    if not text:
        return ""

    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text)
    text = strip_numeric_units(text)
    text = normalize_unicode_math(text)
    text = strip_math_delimiters(text)

    text = text.replace(r"\left", "").replace(r"\right", "")
    text = text.replace(" ", "").lower()

    # Normalise base notation: 4210_{5} -> 4210_5
    text = re.sub(r"_\{([0-9]+)\}", r"_\1", text)

    # Normalise common LaTeX commands
    text = text.replace(r"\dfrac", r"\frac")
    text = text.replace(r"\tfrac", r"\frac")
    text = text.replace(r"\cfrac", r"\frac")

    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\infty", "infty")
    text = text.replace(r"\cup", "cup")
    text = text.replace(r"\cap", "cap")
    text = text.replace(r"\geq", ">=")
    text = text.replace(r"\leq", "<=")
    text = text.replace(r"\ge", ">=")
    text = text.replace(r"\le", "<=")
    text = text.replace(r"\in", "in")

    # Remove common answer prefixes
    text = re.sub(r"^(?:answer|ans|finalanswer|result)=?", "", text)

    # Handle membership form: x\in[-2,7] -> [-2,7]
    text = re.sub(r"^[a-z]+in(?=[\[\(\{])", "", text)

    # Normalize sqrt before flattening fractions
    text = re.sub(r"\\sqrt(?!\{)\s*([a-z0-9]+)", r"\\sqrt{\1}", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)

    # Flatten LaTeX fractions while preserving necessary parentheses:
    # \frac{11+9a}{20} -> ((11+9a)/(20))
    # \frac{\pi}{2} -> ((pi)/(2))
    for _ in range(10):
        new_text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        if new_text == text:
            break
        text = new_text

    # Simplify only atomic fraction parentheses:
    # ((pi)/(2)) -> pi/2
    # ((24)/(25)) -> 24/25
    # Do NOT simplify ((11+9a)/(20))
    for _ in range(5):
        new_text = re.sub(
            r"\(\((-?[a-z0-9^.]+)\)/\((-?[a-z0-9^.]+)\)\)",
            r"\1/\2",
            text,
        )
        if new_text == text:
            break
        text = new_text

    # Critical fix: Sort the additive terms so that 18+2π and 2π+18 are identical after normalisation.
    def sort_additive_terms(expr: str) -> str:
        """ Sort the terms in addition-subtraction expressions to make \(18 + 2\pi\) and \(2\pi + 18\) equivalent. """
        # If the input contains parentheses, equal signs or complex structures, it shall not be processed.
        if any(c in expr for c in '()=<>*/'):
            return expr
        
        # Ensure that the expression starts with either a plus sign (+) or a minus sign (-)
        if not expr.startswith('+') and not expr.startswith('-'):
            expr = '+' + expr
        
        # Split item: split according to + or -, while retaining the symbols.
        terms = []
        current_term = ""
        for i, ch in enumerate(expr):
            if ch in '+-' and i > 0:
                if current_term:
                    terms.append(current_term)
                current_term = ch
            else:
                current_term += ch
        if current_term:
            terms.append(current_term)
        
        if len(terms) <= 1:
            return expr.lstrip('+')
        
        # Classification
        numeric_terms = []
        symbolic_terms = []
        
        for term in terms:
            if not term:
                continue
            
            # Symbol extraction
            sign = 1
            if term.startswith('-'):
                sign = -1
                content = term[1:]
            elif term.startswith('+'):
                sign = 1
                content = term[1:]
            else:
                content = term
            
            # Determine whether the input consists entirely of digits.
            if re.fullmatch(r'\d+\.?\d*', content):
                numeric_terms.append(sign * float(content))
            else:
                # For symbolic terms, the complete term (including the coefficient) shall be retained.
                symbolic_terms.append((content, sign))
        
        # Sorted Symbol Items: Alphabetical Order
        symbolic_terms.sort(key=lambda x: x[0])
        
        # Recombination
        result_parts = []
        
        # Place the numeric items first.
        total_numeric = sum(numeric_terms)
        if total_numeric != 0:
            if total_numeric > 0:
                result_parts.append(f"+{total_numeric}")
            else:
                result_parts.append(f"{total_numeric}")
        
        # Insert the symbol items again.
        for content, sign in symbolic_terms:
            if sign > 0:
                result_parts.append(f"+{content}")
            else:
                result_parts.append(f"-{content}")
        
        if not result_parts:
            return "0"
        
        result = ''.join(result_parts)
        # Remove the leading plus sign.
        if result.startswith('+'):
            result = result[1:]
        
        return result
    
    # Attempt to sort the terms of the expression
    text = sort_additive_terms(text)

    # Remove harmless parentheses only around atomic terms
    for _ in range(5):
        new_text = re.sub(r"\((-?[a-z0-9^.]+)\)", r"\1", text)
        if new_text == text:
            break
        text = new_text

    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    text = text.replace(";", ",")

    return text.strip()
    
    # Attempt to sort the terms of the expression
    text = sort_additive_terms(text)

    # Remove harmless parentheses only around atomic terms
    for _ in range(5):
        new_text = re.sub(r"\((-?[a-z0-9^.]+)\)", r"\1", text)
        if new_text == text:
            break
        text = new_text

    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\", "")
    text = text.replace(";", ",")

    return text.strip()


def normalize_matrix_vector(text: str) -> str:
    """ Normalised Matrix/Vector Representation """
    # Remove \begin{pmatrix} and \end{pmatrix}
    text = re.sub(r'\\begin\{pmatrix\}', '[', text)
    text = re.sub(r'\\end\{pmatrix\}', ']', text)
    # Remove LaTeX line break commands
    text = text.replace(r'\\', ';')
    # Remove spaces and normalise.
    text = re.sub(r'\s+', '', text.strip())
    return text


def looks_like_structured_answer(text: Optional[str]) -> bool:
    if text is None:
        return False
    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text).strip()

    # simple parenthesised single token like (c), (-21) is not a coordinate/structured answer.
    if re.fullmatch(r"\([A-Za-z0-9+\-]+\)", text):
        return False

    if any(marker in text for marker in [r"\cup", r"\cap", r"\in", "∪", "∩", "[", "]"]):
        return True
    if re.fullmatch(r"\([^()]*,[^()]*\)", text):
        return True
    # Matrix/vector detection
    if r"\begin{pmatrix}" in text or r"\begin{bmatrix}" in text:
        return True
    return False


def looks_like_base_notation(text: Optional[str]) -> bool:
    if not text:
        return False

    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text)
    text = strip_math_delimiters(text)
    text = normalize_unicode_math(text)
    text = text.replace(" ", "")

    # Examples: 52_8, 204_5, 4210_5, 1011_2
    return bool(re.fullmatch(r"\$?[-+]?[0-9A-Fa-f]+_\{?[0-9]+\}?\$?", text))


def looks_like_expression_answer(text: Optional[str]) -> bool:
    if text is None:
        return False

    if looks_like_base_notation(text):
        return True

    text = extract_final_answer_text(text)
    text = strip_latex_wrappers(text).strip()

    # Do not treat pure numerical scores as expressions.
    if re.fullmatch(r"\\frac\{[-+]?\d+\}\{[-+]?\d+\}", text):
        return False
    if re.fullmatch(r"\\?frac\s*\{?\s*[-+]?\d+\s*\}?\s*\{?\s*[-+]?\d+\s*\}?", text):
        return False

    # Notation of numeral systems, e.g., 204_5
    if re.search(r"\d+_\d+", text):
        return True

    # Distinct features of mathematical expressions
    if "=" in text:
        return True
    if any(func in text for func in ["\\sin", "\\cos", "\\tan", "\\cot", "\\sec", "\\csc"]):
        return True
    if "\\sqrt" in text or "^" in text:
        return True

    # Only a combination of letters with mathematical symbols, parentheses or numerals constitutes a valid expression.
    if re.search(r"[A-Za-z]", text) and re.search(r"[0-9+\-*/^_=()]", text):
        return True

    return False


def is_pure_numeric_like(text: Optional[str]) -> bool:
    if not text:
        return False

    cleaned = extract_final_answer_text(text)
    cleaned = strip_latex_wrappers(cleaned)
    cleaned = strip_numeric_units(cleaned)
    cleaned = normalize_unicode_math(cleaned)
    cleaned = strip_math_delimiters(cleaned)

    if looks_like_base_notation(cleaned):
        return False

    if parse_numeric_string(cleaned) is not None:
        return True

    if extract_number_from_simple_equation(cleaned) is not None:
        return True

    return False


def detect_answer_type(text: Optional[str]) -> str:
    cleaned = extract_final_answer_text(text)
    cleaned = strip_latex_wrappers(cleaned)

    if not cleaned:
        return "missing"

    if looks_like_base_notation(cleaned):
        return "expression"

    if looks_like_structured_answer(cleaned):
        return "structured"

    if is_pure_numeric_like(cleaned):
        return "numeric"

    if looks_like_expression_answer(cleaned):
        return "expression"

    if re.fullmatch(r"[A-Za-z][A-Za-z\s\-]*", cleaned):
        return "text"

    if re.fullmatch(r"[A-Ea-e]", cleaned):
        return "text"

    return "expression"


def format_answer_display(raw_text: Optional[str]) -> Tuple[str, str]:
    cleaned = extract_final_answer_text(raw_text)
    if not cleaned:
        return ("N/A", "missing")

    answer_type = detect_answer_type(cleaned)
    if answer_type == "numeric":
        numeric_value = extract_number(cleaned)
        return format_numeric_answer(numeric_value)

    display = cleaned if len(cleaned) <= 50 else cleaned[:47] + "..."
    return (display, answer_type)


# Symbolic expression comparison

if HAS_SYMPY:
    def to_sympy_expr(s: str):
        """ Convert the normalized string into a SymPy expression. """
        # Check whether it is suitable for symbolisation.
        if any(token in s for token in ["[", "]", "cup", "cap", "infty", ",", "_"]):
            # For sets/intervals, a specialised comparison is applied.
            return None
        
        # preprocessing
        s = s.replace("^", "**")
        s = re.sub(r"(?<=[0-9\)])i\b", r"*I", s)
        s = re.sub(r"\bi\b", "I", s)
        
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        
        local_dict = {
            "pi": sp.pi,
            "sqrt": sp.sqrt,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "I": sp.I,
            "E": sp.E,
        }
        
        # Create a symbol for the variable.
        for ch in set(re.findall(r'[a-z]', s)):
            if ch not in local_dict and ch != 'i':
                local_dict[ch] = sp.Symbol(ch)
        
        return parse_expr(
            s,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
else:
    def to_sympy_expr(s: str):
        return None


def is_equivalent_expression(model_text: Optional[str], ref_text: Optional[str]) -> bool:
    if model_text is None or ref_text is None:
        return False
    
    model_norm = normalize_expression_string(model_text)
    ref_norm = normalize_expression_string(ref_text)
    
    if not model_norm or not ref_norm:
        return False
    
    # 1.Exact match
    if model_norm == ref_norm:
        return True
    
    # 2.Special handling for base-n notation: 4210_5 vs 5 should return False
    # Check whether one value is represented in a positional numeral system while the other is a common decimal number.
    model_is_base = looks_like_base_notation(str(model_text))
    ref_is_base = looks_like_base_notation(str(ref_text))
    
    if model_is_base != ref_is_base:
        # One entity adopts a binary representation, while the other does not.
        model_num = extract_number(model_text)
        ref_num = extract_number(ref_text)
        if model_num is not None and ref_num is not None:
            # The representation of a number in a given number system should not be equal to its ordinary decimal representation.
            return False
        # If no numerical value can be extracted, the result shall also be deemed as a mismatch.
        return False
    
    # If both parties use binary representation, directly compare the normalised character strings.
    if model_is_base and ref_is_base:
        return model_norm == ref_norm
    
    # 3.Attempt a numerical comparison if the data consists of pure numerical values.
    model_num = extract_number(model_text)
    ref_num = extract_number(ref_text)
    if model_num is not None and ref_num is not None:
        abs_diff = abs(model_num - ref_num)
        rel_diff = abs_diff / (max(abs(ref_num), 1e-9))
        return abs_diff <= ABS_TOL or rel_diff <= REL_TOL
    
    # 4.Attempt to extract the values on the right-hand side of the equation.
    model_rhs = extract_number_from_simple_equation(model_text)
    ref_rhs = extract_number_from_simple_equation(ref_text)
    if model_rhs is not None and ref_rhs is not None:
        abs_diff = abs(model_rhs - ref_rhs)
        rel_diff = abs_diff / (max(abs(ref_rhs), 1e-9))
        return abs_diff <= ABS_TOL or rel_diff <= REL_TOL
    
    # 5.Special Comparisons of Matrices/Vectors
    if r"\begin{pmatrix}" in str(model_text) or r"\begin{pmatrix}" in str(ref_text):
        model_mat = normalize_matrix_vector(str(model_text))
        ref_mat = normalize_matrix_vector(str(ref_text))
        if model_mat == ref_mat:
            return True
    
    # 6.Symbol Equivalence Checking
    if HAS_SYMPY:
        try:
            m_expr = to_sympy_expr(model_norm)
            r_expr = to_sympy_expr(ref_norm)
            
            if m_expr is not None and r_expr is not None:
                diff = sp.simplify(m_expr - r_expr)
                return diff == 0
                
        except Exception as e:
            logger.debug(f"Sympy comparison failed: {str(e)}")
    
    # 7.String Similarity Comparison
    # Remove all spaces and formatting differences.
    model_simple = re.sub(r'\s+', '', model_norm)
    ref_simple = re.sub(r'\s+', '', ref_norm)
    if model_simple == ref_simple:
        return True
    
    return False


def is_answer_correct(model_text: Optional[str], ref_text: Optional[str]) -> bool:
    if not model_text or not ref_text:
        return False
    
    m_raw = extract_rhs_if_equation_chain(model_text)
    r_raw = extract_rhs_if_equation_chain(ref_text)
    
    m_type = detect_answer_type(m_raw)
    r_type = detect_answer_type(r_raw)
    
    # Special Processing: Binary Representation vs. Common Numerals
    m_is_base = looks_like_base_notation(str(m_raw))
    r_is_base = looks_like_base_notation(str(r_raw))
    
    if m_is_base and not r_is_base:
        # The model outputs a base-n representation, while the reference answer is presented in the form of standard decimal digits.
        # For example, the model outputs '4210_5', while the ground truth reference is '5'.
        m_num = extract_number(m_raw)
        r_num = extract_number(r_raw)
        if m_num is not None and r_num is not None:
            # Radical representations should not be equated with conventional numbers.
            return False
        return is_equivalent_expression(m_raw, r_raw)
    
    if not m_is_base and r_is_base:
        # The model outputs ordinary decimal numbers, while the reference answer adopts a positional numeral system representation.
        m_num = extract_number(m_raw)
        r_num = extract_number(r_raw)
        if m_num is not None and r_num is not None:
            return False
        return is_equivalent_expression(m_raw, r_raw)
    
    if m_is_base and r_is_base:
        # Both representations are in positional numeral systems and can be directly compared.
        return normalize_expression_string(m_raw) == normalize_expression_string(r_raw)
    
    # 1.Text answer
    if m_type == "text" and r_type == "text":
        return normalize_text_answer(m_raw) == normalize_text_answer(r_raw)
    
    # 2.Pure numerical responses (or those from which numerical values can be extracted)
    m_num = extract_number(m_raw) if m_type in ("numeric", "expression") else None
    r_num = extract_number(r_raw) if r_type in ("numeric", "expression") else None
    
    if m_num is not None and r_num is not None:
        abs_diff = abs(m_num - r_num)
        rel_diff = abs_diff / (max(abs(r_num), 1e-9))
        if abs_diff <= ABS_TOL or rel_diff <= REL_TOL:
            return True
    
    # 3.If one is explicitly numeric and the other is also numeric but the values do not match.
    if m_type == "numeric" and r_type == "numeric":
        return False  
    
    # 4.Comparison of Expression / structured
    if m_type in {"expression", "structured"} or r_type in {"expression", "structured"}:
        # Try extracting the values for comparison first.
        if m_num is not None and r_num is not None:
            return False  

        # Handle simple assignment-style answers:
        # model: r = \sqrt{5}, reference: \sqrt{5}
        m_rhs = extract_rhs_from_single_variable_equation(m_raw)
        r_rhs = extract_rhs_from_single_variable_equation(r_raw)

        if m_rhs is not None and is_equivalent_expression(m_rhs, r_raw):
            return True

        if r_rhs is not None and is_equivalent_expression(m_raw, r_rhs):
            return True

        if m_rhs is not None and r_rhs is not None and is_equivalent_expression(m_rhs, r_rhs):
            return True

    return is_equivalent_expression(m_raw, r_raw)
    
    # 5.Final fallback
    return normalize_text_answer(m_raw) == normalize_text_answer(r_raw)


def answer_key(text: Optional[str]) -> Optional[str]:
    cleaned = extract_final_answer_text(text)
    if not cleaned:
        return None

    answer_type = detect_answer_type(cleaned)

    if answer_type == "numeric":
        numeric_value = extract_number(cleaned)
        if numeric_value is not None:
            return f"num:{numeric_value:.12g}"

    if answer_type in {"structured", "expression"}:
        return f"expr:{normalize_expression_string(cleaned)}"

    return f"text:{normalize_text_answer(cleaned)}"


def majority_answer(answer_list: List[Optional[str]]) -> Optional[str]:
    valid_answers = [a for a in answer_list if answer_key(a) is not None]
    if not valid_answers:
        return None

    counts = defaultdict(int)
    mapping = {}
    for ans in valid_answers:
        key = answer_key(ans)
        counts[key] += 1
        mapping[key] = ans

    maj_key = max(counts, key=counts.get)
    return mapping[maj_key]


def compute_consistency(answer_list: List[Optional[str]]) -> float:
    """Supplementary only: multi-run agreement across K runs."""
    valid_answers = [a for a in answer_list if answer_key(a) is not None]
    if not valid_answers:
        return 0.0

    maj = majority_answer(valid_answers)
    if maj is None:
        return 0.0

    maj_key = answer_key(maj)
    match_count = sum(1 for ans in answer_list if answer_key(ans) == maj_key)
    return match_count / len(answer_list)


# API call

PROMPTS = {
    "base": (
        "You are an answer-only math solver.\n"
        "Return ONLY the final answer.\n"
        "Do NOT explain.\n"
        "Do NOT write any reasoning steps.\n"
        "Do NOT write phrases such as 'the answer is', 'therefore', 'thus', 'hence', 'so', or 'we get'.\n"
        "Do NOT include full sentences.\n"
        "Return exactly one line.\n"
        "If the answer is a number, return only the number.\n"
        "If the answer is a fraction, radical, expression, equation, interval, set, coordinate pair, matrix, or choice letter, "
        "return only that mathematical object.\n"
        "If there are multiple answers, separate them with commas.\n"
        "Any explanation will be considered incorrect."
    )
}

prompt_name = "base"

# Turn this off if the full run becomes too slow.
ENABLE_CLEANUP = True

retry = retry_decorator()


def looks_like_not_final_only(text: str) -> bool:
    """
    Detect whether the model output looks like an explanation rather than a final answer only.
    This is only used to decide whether to run a second cleanup call.
    """
    if not text:
        return True

    t = text.strip()

    # Multiple lines often indicate reasoning or explanation.
    # But matrices may naturally contain line breaks, so do not treat matrix-only answers as bad.
    if "\n" in t and not (r"\begin{pmatrix}" in t or r"\begin{bmatrix}" in t):
        return True

    bad_patterns = [
        r"(?i)\btherefore\b",
        r"(?i)\bthus\b",
        r"(?i)\bhence\b",
        r"(?i)\bwe get\b",
        r"(?i)\bwe have\b",
        r"(?i)\bthe answer is\b",
        r"(?i)\bthe result is\b",
        r"(?i)\bthe final answer is\b",
        r"(?i)\bsolution\b",
        r"(?i)\bsolving\b",
        r"(?i)\bprimitive\b",
        r"(?i)\broot of\b",
        r"(?i)\bsatisfies\b",
        r"(?i)\bis equal to\b",
        r"(?i)\bequals\b",
    ]

    return any(re.search(p, t) for p in bad_patterns)


@retry_decorator()
def call_gemini_api(problem: str) -> Dict[str, Any]:
    if MOCK_MODE:
        fake_output = "42"
        return {
            "success": True,
            "raw_output": fake_output,
            "error_message": None,
            "attempts_used": 1,
        }

    if not API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is not set. Please set it as an environment variable instead of hard-coding it."
        )

    try:
        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=problem,
            config=types.GenerateContentConfig(
                system_instruction=PROMPTS[prompt_name],
                temperature=0,
                max_output_tokens=1024,
            ),
        )

        content = ""
        if hasattr(response, "text") and response.text:
            content = response.text.strip()

        if not content:
            content = str(response).strip()

        return {
            "success": True,
            "raw_output": content,
            "error_message": None,
            "attempts_used": 1,
        }

    except Timeout as e:
        logger.error(f"Timeout: {str(e)}")
        raise RuntimeError(f"Timeout: {str(e)}")

    except SSLError as e:
        logger.error(f"SSLError: {str(e)}")
        raise RuntimeError(f"SSLError: {str(e)}")

    except ProxyError as e:
        logger.error(f"ProxyError: {str(e)}")
        raise RuntimeError(f"ProxyError: {str(e)}")

    except RequestException as e:
        logger.error(f"RequestException: {str(e)}")
        raise RuntimeError(f"RequestException: {str(e)}")

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {str(e)}\noriginal response: {response.text[:500]}")
        raise RuntimeError(f"JSONDecodeError: {str(e)}\noriginal response: {response.text[:500]}")

    except KeyError as e:
        logger.error(f"API response format error: {str(e)}\nresponse: {response.text[:500]}")
        raise RuntimeError(f"API response format error: {str(e)}\nresponse: {response.text[:500]}")

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise RuntimeError(str(e))


def check_environment():
    required_packages = [
        "requests", "tqdm", "matplotlib", "seaborn", 
        "sympy",  # For Equivalence Checking of Symbolic Expressions
    ]
    optional_packages = [
        "tenacity",  # Optional Retry Library
    ]
    
    logger.info("Environment check is in progress...")
    missing_required = []
    
    for pkg in required_packages:
        try:
            version = metadata.version(pkg)
            logger.info(f"✅ {pkg} {version}")
        except metadata.PackageNotFoundError:
            missing_required.append(pkg)
            logger.error(f"❌ {pkg} is missing")
    
    if missing_required:
        logger.error(f"Missing required packages: {', '.join(missing_required)}")
        print("❌ Environment inspection failed:")
        print(f"Missing required packages: {', '.join(missing_required)}")
        print("\n💡 Run:")
        print(f"pip install {' '.join(missing_required)}")
        sys.exit(1)
    
    for pkg in optional_packages:
        try:
            version = metadata.version(pkg)
            logger.info(f"✅ Optional package {pkg} {version} is installed")
        except metadata.PackageNotFoundError:
            logger.info(f"ℹ️  Optional package {pkg} is not installed (using fallback)")
    
    logger.info("✅ Environmental inspection passed")
    print("✅ Environmental inspection passed")


# Visualisation module

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
        plt.savefig(os.path.join(output_dir, "subject_accuracy.png"), dpi=300)
        plt.close()

    by_level = report["stats"]["by_level"]
    if by_level:
        plt.figure(figsize=(8, 8))
        levels = list(by_level.keys())
        counts = [by_level[lvl]["total"] for lvl in levels]
        if sum(counts) > 0:
            plt.pie(counts, labels=levels, autopct="%1.1f%%", startangle=90)
            plt.title("Difficulty Distribution")
            plt.savefig(os.path.join(output_dir, "level_distribution.png"), dpi=300)
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
        plt.savefig(os.path.join(output_dir, "answer_type_comparison.png"), dpi=300)
        plt.close()

    logger.info(f"Visualization charts have been saved to {output_dir}")
    print(f"\n📈 Visualization charts have been saved to {output_dir}")



# Main evaluation process

def load_dataset(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at line {line_no}: {str(e)}")

    return [item for item in dataset if item.get("problem") and item.get("answer")]


def run_evaluation():
    check_environment()

    stats = {
        "global": {"correct": 0, "total": 0},
        "subjects": StatsCollector(),
        "levels": StatsCollector(),
        "answer_types": {"model": defaultdict(int), "reference": defaultdict(int)},
        "consistency_sum": 0.0,
        "consistency_count": 0,
    }

    results = []
    failed_cases = []

    dataset = load_dataset(DATA_PATH)
    logger.info(f"Loaded {len(dataset)} valid problems from {DATA_PATH}")
    print(f"The amount of valid data after cleaning: {len(dataset)}")

    progress_bar = tqdm(
        dataset,
        desc="Assessment progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [Accuracy: {postfix}]",
    )

    for item in progress_bar:
        try:
            run_outputs = []
            parsed_answers = []

            for run_id in range(K):
                api_result = call_gemini_api(item["problem"])
                raw_response = api_result["raw_output"]
                final_answer_text = extract_final_answer_text(raw_response)
                parsed_numeric = extract_number(final_answer_text)

                run_outputs.append({
                    "run_id": run_id + 1,
                    "success": api_result["success"],
                    "raw_output": raw_response,
                    "final_answer_text": final_answer_text,
                    "parsed_numeric": parsed_numeric,
                    "error_message": api_result["error_message"],
                    "attempts_used": api_result["attempts_used"],
                })
                parsed_answers.append(final_answer_text)

            primary_raw_response = run_outputs[0]["raw_output"]
            model_text = run_outputs[0]["final_answer_text"]
            ref_text = extract_final_answer_text(item["answer"])

            correct = is_answer_correct(model_text, ref_text)

            consistency = compute_consistency(parsed_answers)
            stats["consistency_sum"] += consistency
            stats["consistency_count"] += 1

            model_display, model_type = format_answer_display(model_text)
            ref_display, ref_type = format_answer_display(ref_text)

            model_num = extract_number(model_text) if model_type == "numeric" else None
            ref_num = extract_number(ref_text) if ref_type == "numeric" else None

            record = {
                "problem": item["problem"],
                "reference_text": ref_text,
                "model_text": model_text,
                "model_numeric": model_num,
                "reference_numeric": ref_num,
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
                "consistency": consistency,
            }
            results.append(record)

            stats["global"]["total"] += 1
            stats["global"]["correct"] += int(correct)
            stats["subjects"].update(record["subject"], correct)
            stats["levels"].update(record["level"], correct)
            stats["answer_types"]["model"][record["model_type"]] += 1
            stats["answer_types"]["reference"][record["reference_type"]] += 1

            current_acc = stats["global"]["correct"] / stats["global"]["total"] if stats["global"]["total"] else 0.0
            progress_bar.set_postfix_str(f"{current_acc:.1%}")

            tqdm.write(
                f"{'✅' if correct else '❌'} {item['problem'][:40]}... | "
                f"Reference: {ref_display} vs Model: {model_display}"
            )

            if not correct:
                logger.debug(f"Incorrect: problem='{item['problem'][:50]}...' ref='{ref_display}' model='{model_display}'")

        except Timeout as e:
            error_msg = f"API timeout: {str(e)}"
            logger.error(error_msg)
            tqdm.write(f"🔥 {error_msg}")
            
            failed_record = {
                "problem": item.get("problem", ""),
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "error": error_msg,
                "is_correct": False,
            }
            failed_cases.append(failed_record)
            results.append(failed_record)
            
            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "unknown"), False)
            stats["levels"].update(item.get("level", "unknown"), False)
            
        except ConnectionError as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            tqdm.write(f"🔥 {error_msg}")
            
            failed_record = {
                "problem": item.get("problem", ""),
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "error": error_msg,
                "is_correct": False,
            }
            failed_cases.append(failed_record)
            results.append(failed_record)
            
            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "unknown"), False)
            stats["levels"].update(item.get("level", "unknown"), False)
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON parse error: {str(e)}"
            logger.error(error_msg)
            tqdm.write(f"🔥 {error_msg}")
            
            failed_record = {
                "problem": item.get("problem", ""),
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "error": error_msg,
                "is_correct": False,
            }
            failed_cases.append(failed_record)
            results.append(failed_record)
            
            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "unknown"), False)
            stats["levels"].update(item.get("level", "unknown"), False)
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(error_msg)
            tqdm.write(f"🔥 {error_msg}")
            
            import traceback
            logger.error(traceback.format_exc())

            failed_record = {
                "problem": item.get("problem", ""),
                "subject": item.get("subject", "unknown"),
                "level": item.get("level", "unknown"),
                "error": error_msg,
                "is_correct": False,
            }
            failed_cases.append(failed_record)
            results.append(failed_record)

            stats["global"]["total"] += 1
            stats["subjects"].update(item.get("subject", "unknown"), False)
            stats["levels"].update(item.get("level", "unknown"), False)

            current_acc = stats["global"]["correct"] / stats["global"]["total"] if stats["global"]["total"] else 0.0
        #    progress_bar.set_postfix_str(f"{current_acc:.1%}")

    global_accuracy = stats["global"]["correct"] / stats["global"]["total"] if stats["global"]["total"] else 0.0
    average_consistency = stats["consistency_sum"] / stats["consistency_count"] if stats["consistency_count"] else 0.0

    report = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_size": len(dataset),
            "config": {
                "model": MODEL_NAME,
                "prompt_name": prompt_name,
                "timeout": REQUEST_TIMEOUT,
                "max_retries": MAX_RETRIES,
                "abs_tol": ABS_TOL,
                "rel_tol": REL_TOL,
                "K": K,
                "primary_metric": "first_pass_final_answer_accuracy",
                "consistency_metric": "supplementary_multi_run_consistency",
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
    logger.info(f"Report saved to {OUTPUT_PATH}")

    save_jsonl(DETAILS_PATH, results)
    logger.info(f"Detailed results saved to {DETAILS_PATH}")
    
    save_jsonl(FAILED_PATH, failed_cases)
    logger.info(f"Failed cases saved to {FAILED_PATH}")
    
    generate_visualizations(report, OUTPUT_DIR)

    print("\n📊 Assessment report")
    print(f"Total: {report['metadata']['data_size']}")
    print(f"Total Accuracy: {report['stats']['global_accuracy']:.2%}")
    print(f"Average Consistency (K={K}): {report['stats']['average_consistency']:.4f}")
    print(f"Failed cases: {len(failed_cases)}")
    
    logger.info(f"Evaluation completed. Accuracy: {global_accuracy:.2%}, Failed: {len(failed_cases)}")

    return report



# Test module

def test_extractor():
    test_cases = [
        ("42", 42.0),
        ("3.1416", 3.1416),
        ("1.23e5", 123000.0),
        ("11/2", 5.5),
        (r"\boxed{11/2}", 5.5),
        (r"\frac{11}{2}", 5.5),
        (r"\boxed{\frac{11}{2}}", 5.5),
        ("5%", 0.05),
        ("50\\%", 0.5),
        ("answer: 42", 42.0),
        (r"final answer: \boxed{3.14}", 3.14),
        ("The answer is 42.", 42.0),
        ("最后数值是42", 42.0),
        ("1,200%", 12.0),
        ("1200.0", 1200.0),
        (r"15\mbox{ cm}^2", 15.0),
        (r"50^\circ", 50.0),
        ("无效答案", None),
    ]

    print("\n🔬 Parsing test is currently in progress...")
    try:
        print(f"Test environment version: tenacity {metadata.version('tenacity')}")
    except Exception:
        pass

    all_passed = True
    for text, expected in test_cases:
        result = extract_number(text)
        if result is None or expected is None:
            success = (result == expected)
        else:
            success = abs(result - expected) < 1e-6
        all_passed = all_passed and success

        display, _ = format_numeric_answer(result)
        exp_display, _ = format_numeric_answer(expected)
        status = "✅" if success else "❌"
        print(f"{status} {text} → {display} (Expected: {exp_display})")

    if not all_passed:
        raise AssertionError("Some parsing tests failed.")
    print("🎉 All parsing tests passed.")


def test_answer_correctness():
    """ Test whether the extraction method is correct. """
    correctness_cases = [
        (r"\boxed{11/2}", "5.5", True),
        (r"15\mbox{ cm}^2", "15", True),
        (r"50^\circ", "50", True),
        (r"\frac{1}{2}", "0.5", True),
        ("Evelyn", "Evelyn", True),
        ("Angela", "Evelyn", False),
        ("(2,3)", "(2, 3)", True),
        (r"\left( 3, \frac{\pi}{2} \right)", "(3, π/2)", True),
        (r"x \in [-2,7]", "[-2, 7]", True),
        (r"-\frac{24}{25}", "-0.96", True),
        (r"r = 4", "4", True),
        (r"x = 5", "5", True),
        (r"18+2\pi", r"2\pi+18", True),
        (r"\frac{11+9a}{20}", r"\frac{9a+11}{20}", True),
        (r"4210_{5}", r"4210_5", True),
        (r"4210_5", "5", False),
        (r"204_5", r"204_5", True),
        (r"2516_8", "2516", False),
        (r"3\sqrt{13}", "3√13", True),
        (r"11\sqrt2", "11√2", True),
        (r"\sqrt{53}", "√53", True),
        (r"2516_8", "2516₈", True),
        (r"\$32,\!348", "32348", True),
        (r"$78", "78", True),
        (r"\$18.90", "18.9", True),
        (r"137 \frac{1}{2}", "137.5", True),
        ("42", "41", False),
        (r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}", r"\begin{pmatrix} -1 & 0 \\ 0 & -1 \end{pmatrix}", True),
    ]

    print("\n🔬 Correctness test is currently in progress...")
    all_passed = True
    for model, ref, expected in correctness_cases:
        result = is_answer_correct(model, ref)
        success = result == expected
        all_passed = all_passed and success
        status = "✅" if success else "❌"
        print(f"{status} model={model[:30]}... | ref={ref[:30]}... → {result} (Expected: {expected})")

    if not all_passed:
        raise AssertionError("Some correctness tests failed.")
    print("🎉 All correctness tests passed.")


if __name__ == "__main__":
    test_extractor()
    test_answer_correctness()

    # Use --test to run only the local parser tests.
    if "--test" not in sys.argv:
        run_evaluation()
