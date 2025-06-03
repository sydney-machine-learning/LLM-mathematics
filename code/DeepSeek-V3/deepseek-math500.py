import json
import re
import sys
import time
import os
import requests
from tqdm import tqdm
from collections import defaultdict
import importlib.metadata as metadata
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


DEEPSEEK_API_KEY = "sk-da0025c4e3f84de082271474bd734f96"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"
DATA_PATH = "datsets/math500.jsonl"
OUTPUT_PATH = "evaluation_results.json"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 45
ERROR_MARGIN = 0.0001


class StatsCollector:
    def __init__(self):
        self.data = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    def update(self, key: str, is_correct: bool) -> None:
        self.data[key]['total'] += 1
        if is_correct:
            self.data[key]['correct'] += 1
    
    def get_accuracy(self, key: str) -> float:
        return self.data[key]['correct'] / self.data[key]['total'] if self.data[key]['total'] else 0.0

def retry_decorator():
    from tenacity import retry, stop_after_attempt, wait_exponential
    return retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=5, max=30)
    )

@retry_decorator()
def call_deepseek_api(problem: str) -> str:
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "MathEvaluator/2.0"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": f"please give numeric answer directly, question：{problem}"
        }],
        "temperature": 0.1,
        "max_tokens": 50
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
            raise requests.HTTPError(f": {json.dumps(error_info, indent=2)}")
            
        return response.json()['choices'][0]['message']['content']
        
    except requests.exceptions.SSLError as e:
        raise RuntimeError(f"SSL证书错误: {str(e)}")
    except requests.exceptions.ProxyError as e:
        raise RuntimeError(f"代理错误: {str(e)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"响应解析失败: {str(e)}\n original response: {response.text[:500]}")

def format_answer(value: Optional[float]) -> Tuple[str, str]:
    if value is None:
        return ("N/A", "missing")
    try:
        if abs(value) >= 1e6 or (0 < abs(value) <= 1e-4):
            return (f"{value:.4e}", "scientific")
        return (f"{value:.6g}", "normal")
    except Exception as e:
        return (f"Invalid ({str(e)})", "error")

def extract_number(text: str) -> Optional[float]:
    original_text = text
    text = text.replace('$', '').replace(' ', '').replace(',', '')
    
    patterns = [
        (r'([-+]?\d*\.?\d+[eE][-+]?\d+)', 'scientific'),
        (r'\\boxed{([^{}]+)}', 'latex'),
        (r'(\d+\.?\d*%)', 'percentage'),
        (r'≈\s*([\d.]+)', 'approximate'),
        (r'([-+]?\d+/\d+)', 'fraction'),
        (r'final result[：:\s]*([-+±]?\d*\.?\d+)', 'chinese'),
        (r'answer[\s:]*([-+±]?\d*\.?\d+)', 'english')
    ]
    
    for pattern, ptype in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            value = matches[-1][0] if isinstance(matches[-1], tuple) else matches[-1]

            try:
                if ptype == 'scientific':
                    return float(value)
                elif ptype == 'percentage':
                    return float(value.strip('%')) / 100
                elif ptype == 'fraction':
                    numerator, denominator = map(float, value.split('/'))
                    return numerator / denominator
                else:
                    return float(value)
            except (ValueError, ZeroDivisionError):
                continue
    
    try:
        numbers = re.findall(r'[-+]?\d*\.?\d+', text)
        if numbers:
            return float(numbers[-1])
    except:
        pass
    
    return None

def is_answer_correct(model_num: Optional[float], ref_num: Optional[float]) -> bool:
    try:
        if None in (model_num, ref_num):
            return False
            
        model = float(model_num)
        ref = float(ref_num)
        
        if model == ref == 0:
            return True
            
        abs_diff = abs(model - ref)
        rel_diff = abs_diff / (abs(ref) + 1e-9)
        return abs_diff <= ERROR_MARGIN or rel_diff <= ERROR_MARGIN
    except (TypeError, ValueError):
        return False

def check_environment():
    required_packages = {
        'tenacity': '8.2.0',
        'requests': '2.32.3',
        'tqdm': '4.67.1',
        'setuptools': '68.2.2',
        'matplotlib': '3.8.2',
        'seaborn': '0.13.2'
    }
    
    print("🔍 正在执行环境检查...")
    missing_packages = []
    version_mismatch = []
    
    for pkg, required_ver in required_packages.items():
        try:
            installed_ver = metadata.version(pkg)
            if installed_ver != required_ver:
                version_mismatch.append(f"{pkg} 需要 {required_ver}，当前 {installed_ver}")
        except metadata.PackageNotFoundError:
            missing_packages.append(pkg)
    
    if missing_packages or version_mismatch:
        print("❌ 环境检查失败：")
        if missing_packages:
            print(f"未安装的包: {', '.join(missing_packages)}")
        if version_mismatch:
            print(f"版本不匹配:\n  - " + "\n  - ".join(version_mismatch))
        print("\n💡 解决方案: 执行以下命令安装依赖")
        print("pip install \\")
        print(f"  tenacity=={required_packages['tenacity']} \\")
        print(f"  requests=={required_packages['requests']} \\")
        print(f"  tqdm=={required_packages['tqdm']} \\")
        print(f"  setuptools=={required_packages['setuptools']} \\")
        print(f"  matplotlib=={required_packages['matplotlib']} \\")
        print(f"  seaborn=={required_packages['seaborn']}")
        sys.exit(1)
    
    print("✅ 环境检查通过")

# ======================
# 可视化模块
# ======================
def generate_visualizations(report: dict, output_dir: str = "visualization_results"):
    """生成可视化图表"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_theme(style="whitegrid", palette="husl")

    # 学科准确率柱状图
    plt.figure(figsize=(12, 6))
    subjects = list(report['stats']['by_subject'].keys())
    accuracies = [v*100 for v in report['stats']['by_subject'].values()]
    sns.barplot(x=subjects, y=accuracies)
    plt.title('学科准确率对比')
    plt.ylabel('准确率 (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/subject_accuracy.png', dpi=300)
    plt.close()

    # 难度分布饼图
    plt.figure(figsize=(8, 8))
    levels = list(report['stats']['by_level'].keys())
    counts = [report['stats']['by_level'][lvl]['total'] for lvl in levels]
    plt.pie(counts, labels=levels, autopct='%1.1f%%', startangle=90)
    plt.title('题目难度分布')
    plt.savefig(f'{output_dir}/level_distribution.png', dpi=300)
    plt.close()

    # 答案类型分布对比
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    model_types = report['stats']['answer_types']['model']
    ref_types = report['stats']['answer_types']['reference']
    
    sns.barplot(x=list(model_types.keys()), y=list(model_types.values()), ax=ax[0])
    ax[0].set_title('模型答案类型分布')
    ax[0].set_ylabel('数量')
    
    sns.barplot(x=list(ref_types.keys()), y=list(ref_types.values()), ax=ax[1])
    ax[1].set_title('参考答案类型分布')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/answer_type_comparison.png', dpi=300)
    plt.close()

    print(f"\n📈 可视化图表已保存至 {output_dir} 目录")

# ======================
# 主评估流程
# ======================
def run_evaluation():
    check_environment()
    
    stats = {
        'global': {'correct': 0, 'total': 0},
        'subjects': StatsCollector(),
        'levels': StatsCollector(),
        'answer_types': {'model': defaultdict(int), 'reference': defaultdict(int)}
    }
    results = []
    
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = [json.loads(line) for line in f if line.strip()]
    
    # 新增数据清洗
    dataset = [item for item in dataset if item.get('problem') and item.get('answer')]
    print(f"清洗后有效数据量：{len(dataset)}")

    progress_bar = tqdm(dataset, desc="🔧 评估进度", 
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [准确率: {postfix}]")

    for item in progress_bar:
        try:
            raw_response = call_deepseek_api(item['problem'])
            model_answer = extract_number(raw_response)
            ref_answer = extract_number(item['answer'])
            correct = is_answer_correct(model_answer, ref_answer)
            
            # 格式化显示
            model_display, model_type = format_answer(model_answer)
            ref_display, ref_type = format_answer(ref_answer)
            
            # 记录结果（统一字段名）
            record = {
                'problem': item['problem'],
                'model_raw': model_answer,
                'reference_raw': ref_answer,
                'model_display': model_display,
                'reference_display': ref_display,
                'model_type': model_type,
                'reference_type': ref_type,
                'is_correct': correct,  # 修改字段名
                'subject': item.get('subject', 'unknown'),
                'level': item.get('level', 'unknown'),
                'raw_response': raw_response[:200] + '...'
            }
            results.append(record)
            
            # 更新统计
            stats['global']['total'] += 1
            stats['global']['correct'] += int(correct)
            stats['subjects'].update(record['subject'], correct)
            stats['levels'].update(record['level'], correct)
            stats['answer_types']['model'][record['model_type']] += 1
            stats['answer_types']['reference'][record['reference_type']] += 1
            
            # 实时显示
            current_acc = stats['global']['correct'] / stats['global']['total']
            progress_bar.set_postfix_str(f"{current_acc:.1%}")
            tqdm.write(
                f"{'✅' if correct else '❌'} {item['problem'][:40]}... | "
                f"参考: {ref_display} vs 模型: {model_display}"
            )
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            tqdm.write(f"🔥 {error_msg}")
            results.append({
                'problem': item['problem'],
                'error': error_msg,
                'is_correct': False  # 统一字段名
            })

    # 生成报告
    report = {
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_size': len(dataset),
            'config': {
                'model': MODEL_NAME,
                'timeout': REQUEST_TIMEOUT,
                'error_margin': ERROR_MARGIN
            }
        },
        'stats': {
            'global_accuracy': stats['global']['correct'] / stats['global']['total'],
            'by_subject': {s: stats['subjects'].get_accuracy(s) for s in stats['subjects'].data},
            'by_level': {l: stats['levels'].get_accuracy(l) for l in stats['levels'].data},
            'answer_types': dict(stats['answer_types'])
        },
        'details': results
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # 生成可视化图表
    generate_visualizations(report)
    
    print("\n📊 评估报告")
    print(f"总题数: {report['metadata']['data_size']}")
    print(f"总准确率: {report['stats']['global_accuracy']:.2%}")
    
    return report

# ======================
# 测试模块
# ======================
def test_extractor():
    """解析器测试"""
    test_cases = [
        ("答案是\\boxed{3.14}", 3.14),
        ("结果约5%", 0.05),
        ("1.23e5", 123000.0),
        ("答案：≈3.1416", 3.1416),
        ("11/2", 5.5),
        ("\\mathbf{6.022e23}", 6.022e23),
        ("无效答案", None),
        ("最后数值是42", 42.0)
    ]
    
    print("\n🔬 正在执行解析测试...")
    print(f"测试环境版本: tenacity {metadata.version('tenacity')}")
    
    for text, expected in test_cases:
        result = extract_number(text)
        success = abs(result - expected) < 1e-6 if (result and expected) else result == expected
        display, _ = format_answer(result)
        exp_display, _ = format_answer(expected)
        status = "✅" if success else "❌"
        print(f"{status} {text} → {display} (预期: {exp_display})")
    
    print("🎉 所有解析测试完成")

if __name__ == "__main__":
    test_extractor()
    run_evaluation()
