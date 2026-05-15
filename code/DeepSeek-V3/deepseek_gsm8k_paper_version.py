import json
import openai
import re
import time
import matplotlib.pyplot as plt

# Configure DeepSeek API
API_KEY = "api"  # Replace with your API key
client = openai.OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"
)

# Load data
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

train_data = load_jsonl(r"datasets/gsm8k.jsonl")  # Path to dataset

# Clean numeric string
def clean_number(num_str):
    num_str = re.sub(r"[,%\s]", "", str(num_str))
    try:
        num = float(num_str)
        return str(int(num)) if num.is_integer() else f"{num:.2f}".rstrip('0').rstrip('.')
    except:
        return num_str

# Call model with retry logic
def solve_math_problem(question, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "Strictly follow these rules: 1. Return only a pure number 2. Keep two decimal places for decimals 3. No units or text"},
                    {"role": "user", "content": question}
                ],
                temperature=0.1
            )
            answer = response.choices[0].message.content
            numbers = re.findall(r"-?\d+\.?\d*", answer)
            return clean_number(numbers[-1]) if numbers else "N/A"
        except Exception as e:
            print(f"Retry {attempt + 1}, error: {str(e)}")
            time.sleep(2)
    return "❌ Exceeded max retries"

# Main testing workflow
total = 7473  # Number of test samples (full GSM8K test set)
correct_count = 0
error_log = []

for i in range(total):
    try:
        question = train_data[i]["question"]
        correct_answer = train_data[i]["answer"].split("#### ")[-1].strip()  # Adapt to GSM8K format
        
        pred_answer = solve_math_problem(question)
        correct_clean = clean_number(correct_answer)
        
        if pred_answer == correct_clean:
            correct_count += 1
        else:
            error_log.append({
                "Question #": i + 1,
                "Question": question,
                "Correct Answer": correct_clean,
                "Model Answer": pred_answer
            })
            print(f"❌ Question {i + 1} incorrect")
        
        time.sleep(0.5)  # Throttle requests
        
    except Exception as e:
        print(f"Data processing error: {str(e)}")

# Output results
accuracy = (correct_count / total) * 100
print(f"\n✅ Final accuracy: {accuracy:.2f}%")

# Save error log
with open("error_log.json", "w", encoding="utf-8") as f:
    json.dump(error_log, f, ensure_ascii=False, indent=2)

# Visualization
plt.figure(figsize=(10, 6))
plt.bar(["Correct", "Incorrect"], [correct_count, total - correct_count], color=["#4CAF50", "#F44336"])
plt.title(f"DeepSeek GSM8K Test Results ({total} Questions)")
plt.ylabel("Number of Questions")
plt.savefig("gsm8k_result.png", dpi=300)
plt.show()
