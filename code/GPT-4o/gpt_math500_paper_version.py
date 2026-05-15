import json
import openai
import os
import re
import sympy as sp
from collections import defaultdict

# ✅ 加载 API key
API_KEY = ""
client = openai.Client(api_key=API_KEY)

# ✅ 读取 math500 数据
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

data = load_jsonl("math500.jsonl")

# ✅ 清洗和提取函数
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

# ✅ GPT 调用
def solve_problem(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            max_tokens=100,
            messages=[
                {"role": "system", "content": """You are a rigorous math solver.

Your task:
- Solve the math problem below.
- Only return the **final answer** in LaTeX, wrapped inside \boxed{...}.
- DO NOT output any explanation, steps, or reasoning.
- If you cannot solve the problem, return \boxed{?}.
- Your output must be a single valid \boxed{...} expression — nothing else.

Example of valid output: \boxed{\frac{1}{2}}"""}, 
                {"role": "user", "content": prompt}
            ]
        )
        raw = response.choices[0].message.content.strip()
        return extract_boxed(raw)
    except Exception as e:
        return f"❌ API Error: {str(e)}"

# ✅ 主评估逻辑
N = 500  # 可调：评估前 N 题
correct = 0
evaluated = 0

level_stats = defaultdict(lambda: {"correct": 0, "total": 0})
subject_stats = defaultdict(lambda: {"correct": 0, "total": 0})

print("\n====== ❌ Incorrect Answers ======\n")

for i in range(N):
    item = data[i]
    q = item["problem"]
    ref = clean_latex(item["answer"])
    level = item.get("level", "unknown")
    subject = item.get("subject", "unknown")

    pred_raw = solve_problem(q)
    pred = clean_latex(pred_raw)

    if "❌" in pred_raw or not is_valid_latex_fraction(pred):
        continue  # 跳过格式错误的输出

    evaluated += 1
    level_stats[level]["total"] += 1
    subject_stats[subject]["total"] += 1

    if is_equivalent(pred, ref):
        correct += 1
        level_stats[level]["correct"] += 1
        subject_stats[subject]["correct"] += 1
    else:
        print(f"❌ Question {i+1} incorrect")
        print(f"✅ Correct answer: {ref}")
        print(f"❌ GPT generated: {pred}\n")

# ✅ 输出总准确率
accuracy = 100 * correct / evaluated if evaluated > 0 else 0
print(f"✅ Total correct: {correct} / {evaluated}")
print(f"✅ GPT-4o total accuracy: {accuracy:.2f}%\n")

# ✅ 分类准确率输出
def print_accuracy(title, stats):
    print(f"📊 {title} Accuracy:")
    for key, v in sorted(stats.items()):
        acc = 100 * v["correct"] / v["total"] if v["total"] > 0 else 0.0
        print(f" - {key}: {acc:.2f}%")
    print()

print_accuracy("Level-wise", level_stats)
print_accuracy("Subject-wise", subject_stats)
