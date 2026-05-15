import json
import openai
import re
import time
import random
from pathlib import Path
from collections import Counter


# Configure DeepSeek API
API_KEY = "api"  # Replace with your API key
client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)

DATA_PATH = "datasets/gsm8k.jsonl"
OUTPUT_DIR = Path("results/deepseek_gsm8k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Main evaluation setting
total = 5000
RANDOM_SEED = 42

#Consistency setting
K = 3

#API robustness setting
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2
REQUEST_TIMEOUT = 30

# Load data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(file_path, rows):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
train_data = load_jsonl(DATA_PATH)  # Path to dataset

def sample_data(data, total, seed=42):
    """Fixed random subset for reproducibility"""
    if total >= len(data):
        return data, list(range(len(data)))

    random.seed(seed)
    indices = sorted(random.sample(range(len(data)), total))
    sampled_data = [data[i] for i in indices]
    return sampled_data, indices
    
# Clean numeric string
def clean_number(num_str):
    if not num_str:
        return None
    try:
        num_str = re.sub(r"[,%\s]", "", str(num_str))
        num = float(num_str)
        return str(int(num)) if num.is_integer() else str(num)
    except:
        return str(num_str)

def extract_numeric_answer(text):
    if not text:
        return None
    numbers = re.findall(r"-?\d+\.?\d*", str(text))
    return clean_number(numbers[-1]) if numbers else None

# Call model with retry logic
def solve_math_problem(question):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", 
                     "content": "You are a math expert. Provide only the numeric answer with no explanation, no units, no commas, no percent signs, and no additional text."
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.1,
                timeout=REQUEST_TIMEOUT
            )
            
            answer = response.choices[0].message.content.strip()
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            
            return {
                "success": True,
                "raw_output": answer,
                "parsed_answer": clean_number(numbers[-1]) if numbers else answer,
                "error_message": None,
                "attempt_used": attempt
            }
            
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"Retry {attempt}, error: {str(e)}")
                time.sleep(RETRY_WAIT_SECONDS)
            else:
                return{
                    "success": False,
                    "raw_output": None,
                    "parsed_answer": None,
                    "error_message": str(e),
                    "attempt_used": attempt
                }

def check_answer(pred_answer, correct_answer):
    correct_number = extract_numeric_answer(correct_answer)
    return pred_answer == correct_number

def majority_answer(answer_list):
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return None
    return Counter(valid_answers).most_common(1)[0][0]

def compute_consistency(answer_list):
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return 0.0
    maj = majority_answer(valid_answers)
    return sum(1 for a in answer_list if a ==maj) / len(answer_list)

#Fixed subset
train_data, sampled_indices = sample_data(train_data, total, RANDOM_SEED)

with open(OUTPUT_DIR / "gsm8k_subset_indices_seed42.json", "w", encoding="utf-8") as f:
    json.dump(sampled_indices, f, ensure_ascii=False, indent=2)


# Main testing workflow
correct_count = 0
successful_count = 0
consistency_sum = 0.0

results =[]
failed_cases = []

for i in range(total):
        question = train_data[i]["question"]
        correct_answer = train_data[i]["answer"]

        run_outputs = []
        parsed_answers = []
        had_failed_run = False
        error_messages = []

        for k in range(K):
            result = solve_math_problem(question)
            run_outputs.append(result)

            if result["success"]:
                parsed_answers.append(result["parsed_answer"])
            else:
                parsed_answers.append(None)
                had_failed_run = True
                error_messages.append(result["error_message"])

        final_answer = majority_answer(parsed_answers)
        consistency = compute_consistency(parsed_answers)
        is_correct = check_answer(final_answer, correct_answer) if final_answer is not None else False

        if final_answer is not None:
            successful_count += 1
            consistency_sum += consistency
            if is_correct:
                correct_count += 1
            else:
                print(f"❌ Question {i + 1} incorrect")
                print(f"✅️ Correct answer: {extract_numeric_answer(correct_answer)}")
                print(f"❌️ DeepSeek answer: {final_answer}\n")
                
        if had_failed_run:
            failed_cases.append({
                "index": i + 1,
                "question": question,
                "error_messages": error_messages
            })
            
        results.append({
            "index": i + 1,
            "question": question,
            "correct_answer_raw": correct_answer,
            "correct_answer_numeric": extract_numeric_answer(correct_answer),
            "runs": run_outputs,
            "parsed_answers": parsed_answers,
            "majority_answer": final_answer,
            "consistency": consistency,
            "is_correct": is_correct,
            "had_failed_run": had_failed_run
        })

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total} questions...")
                

# Output results
accuracy = (correct_count / successful_count) * 100 if successful_count > 0 else 0
average_consistency = consistency_sum / successful_count if successful_count >0 else 0

summary = {
    "model": "deepseek-chat",
    "dataset": "GSM8K",
    "total_requested": total,
    "total_actual": len(train_data),
    "sampling_method": "fixed_random_subset",
    "random_seed": RANDOM_SEED,
    "answer_only_prompt": True,
    "K": K,
    "max_retries": MAX_RETRIES,
    "timeout_seconds": REQUEST_TIMEOUT,
    "successful_count": successful_count,
    "failed_count": len(failed_cases),
    "accuracy_percent": round(accuracy,2),
    "average_consistency": round(average_consistency,4),
    "correct_count": correct_count
}


save_jsonl(OUTPUT_DIR / "gsm8k_deepseek_results.jsonl", results)
save_jsonl(OUTPUT_DIR / "gsm8k_deepseek_failed_cases.jsonl", failed_cases)

with open(OUTPUT_DIR / "gsm8k_deepseek_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✅ Final accuracy: {accuracy:.2f}%")
print(f"✅️ Average consistency (K={K}): {average_consistency:.4f}")
print(f"⚠️ Failed cases: {len(failed_cases)}")

