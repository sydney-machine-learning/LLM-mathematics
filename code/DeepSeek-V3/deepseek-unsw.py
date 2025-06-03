import pandas as pd
import requests
import json
import logging
from tqdm import tqdm

# 配置
API_KEY = "sk-da0025c4e3f84de082271474bd734f96"
CSV_PATH = "datasets/UNSW-problems/unsw.csv"
OUTPUT_CSV = "results.csv"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 设置日志
logging.basicConfig(filename='api_errors.log', level=logging.ERROR)

def call_deepseek_api(question):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-math-v1",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.1
    }
    try:
        response = requests.post(API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logging.error(f"Error processing question: {question}\nError: {str(e)}")
        return None

def evaluate_answer(model_ans, ref_ans):
    # 简单文本对比（复杂场景需定制逻辑）
    return model_ans.strip() == ref_ans.strip()

def main():
    # 读取CSV
    df = pd.read_csv(CSV_PATH)
    results = []
    
    # 遍历每个问题
    for _, row in tqdm(df.iterrows(), total=len(df)):
        question = f"{row['question_text']}\nReference Answer: {row['reference_answer']}"
        model_answer = call_deepseek_api(question)
        
        if model_answer:
            is_correct = evaluate_answer(model_answer, row['reference_answer'])
        else:
            is_correct = False
        
        results.append({
            "problem_id": row['problem_id'],
            "sub_part": row['sub_part'],
            "model_answer": model_answer,
            "is_correct": is_correct
        })
    
    # 保存结果
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
