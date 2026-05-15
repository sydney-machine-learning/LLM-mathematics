import json
import re
import time
import random
from google import genai
from google.genai import types
from pathlib import Path
from collections import Counter


# Gemini API setting
API_KEY = " "  # Set API key before running
client = genai.Client(api_key=API_KEY)
MODEL_NAME = "gemini-2.5-flash" 


DATA_PATH = "datasets/gsm8k.jsonl"
OUTPUT_DIR = Path("results/gemini_gsm8k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Main evaluation setting
total = 5000
RANDOM_SEED = 42

# Number of repeated runs per question
K = 3

# Retry settings
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 8

BASE_PROMPT = ("You are a math expert. Provide only the numeric answer with no explanation, no units, no commas, no percent signs, and no additional text."
              "Question:{question}")

# Data Loading
def load_jsonl(file_path):
    """ Load a JSONL file. """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(file_path, rows):
    """ Save rows to a JSONL file. """
    with open(file_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
train_data = load_jsonl(DATA_PATH)  # Load GSM8K dataset

def sample_data(data, total, seed=42):
    """ Select a fixed random subset for reproducibility. """
    """Fixed random subset for reproducibility"""
    if total >= len(data):
        return data, list(range(len(data)))

    random.seed(seed)
    indices = sorted(random.sample(range(len(data)), total))
    sampled_data = [data[i] for i in indices]
    return sampled_data, indices
    
# Answer parsing utilities
def clean_number(num_str):
    """ Normalise a numeric string for somparison. """
    if not num_str:
        return None
    try:
        num_str = re.sub(r"[,%\s]", "", str(num_str))
        num = float(num_str)
        return str(int(num)) if num.is_integer() else str(num)
    except:
        return str(num_str)

def extract_numeric_answer(text):
    """ Extract the last numeric value from an answer text. """
    if not text:
        return None
    text = str(text).replace(",","")
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return clean_number(numbers[-1]) if numbers else None

# Gemini model call
def solve_math_problem(question):
    """ Generate a numeric answer using the selected Gemini model. """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            prompt = BASE_PROMPT.format(question=question)
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.1
                )
            )
               
            answer = response.text.strip()
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
    """ Compare the parsed model answer with the numeric reference answer. """
    correct_number = extract_numeric_answer(correct_answer)
    return pred_answer == correct_number

def majority_answer(answer_list):
    """ Return the most frequent valid answer. """
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return None
    return Counter(valid_answers).most_common(1)[0][0]

def compute_consistency(answer_list):
    """ Compute the proportion of answers matching the majority answer. """
    valid_answers = [a for a in answer_list if a is not None]
    if not valid_answers:
        return 0.0
    maj = majority_answer(valid_answers)
    return sum(1 for a in answer_list if a ==maj) / len(answer_list)

# Select fixed subset for reproducibility
train_data, sampled_indices = sample_data(train_data, total, RANDOM_SEED)

with open(OUTPUT_DIR / "gsm8k_subset_indices_seed42.json", "w", encoding="utf-8") as f:
    json.dump(sampled_indices, f, ensure_ascii=False, indent=2)


# Main evaluation loop
correct_count = 0
successful_count = 0
consistency_sum = 0.0

results =[]
failed_cases = []

for i in range(len(train_data)):
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
                print(f"❌️ Gemini answer: {final_answer}\n")
                
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
                

# Output results and summary
accuracy = (correct_count / successful_count) * 100 if successful_count > 0 else 0
average_consistency = consistency_sum / successful_count if successful_count >0 else 0

summary = {
    "model": "gemini-2.5-flash",
    "dataset": "GSM8K",
    "total_requested": total,
    "total_actual": len(train_data),
    "sampling_method": "fixed_random_subset",
    "random_seed": RANDOM_SEED,
    "answer_only_prompt": True,
    "K": K,
    "max_retries": MAX_RETRIES,
    "successful_count": successful_count,
    "failed_count": len(failed_cases),
    "accuracy_percent": round(accuracy,2),
    "average_consistency": round(average_consistency,4),
    "correct_count": correct_count
}


save_jsonl(OUTPUT_DIR / "gsm8k_gemini_results.jsonl", results)
save_jsonl(OUTPUT_DIR / "gsm8k_gemini_failed_cases.jsonl", failed_cases)

with open(OUTPUT_DIR / "gsm8k_gemini_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n✅ Final accuracy: {accuracy:.2f}%")
print(f"✅️ Average consistency (K={K}): {average_consistency:.4f}")
print(f"⚠️ Failed cases: {len(failed_cases)}")
