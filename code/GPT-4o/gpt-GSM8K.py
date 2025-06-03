
import json
import openai  # Change model
import re

API_KEY = "#api key"
client = openai.Client(api_key=API_KEY)

def load_jsonl(file_path):
    """ Read JSONL data line by line """
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Load training data
train_data = load_jsonl("datasets/gsm8k.jsonl")

def clean_number(num_str):
    """ Clean numeric string: remove commas, spaces, percent signs, and convert to int or float """
    if not num_str:
        return None
    try:
        num_str = re.sub(r"[,%\s]", "", num_str)  # Remove commas, spaces, percent signs
        num = float(num_str)  # Convert to float
        return str(int(num)) if num.is_integer() else str(num)  # Format
    except ValueError:
        return num_str  # Return original if conversion fails

def extract_numeric_answer(text):
    """ Extract the numeric value from the answer text """
    numbers = re.findall(r"-?\d+\.?\d*", text)  # Match integers and decimals
    return clean_number(numbers[-1]) if numbers else None  # Take the last number

def solve_math_problem(question):
    """ Ask GPT-4o to return only the final numeric answer (no units, commas, percent signs, or extra zeros) """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Change model
            messages=[
                {"role": "system", "content": "You are a math expert. Provide only the numeric answer with no explanation, no units, no commas, no percent signs, and no additional text."},
                {"role": "user", "content": question}
            ]
        )
        gpt_answer = response.choices[0].message.content.strip()

        # 🔹 Keep only the numeric part of GPT-4o's response, remove units, commas, percent signs, and extra zeros
        numbers = re.findall(r"-?\d+\.?\d*", gpt_answer)  # Extract numbers
        return clean_number(numbers[-1]) if numbers else gpt_answer  # Format result

    except Exception as e:
        return f"❌ API error: {str(e)}"

def check_answer(gpt_answer, correct_answer):
    """ Compare GPT's answer with the ground truth (strip units, commas, percent signs, and extra zeros) """
    correct_number = extract_numeric_answer(correct_answer)  # Extract numeric value from correct answer
    return gpt_answer == correct_number  # Compare only numeric parts

# Evaluate GPT-4o on the first 500 math problems
correct_count = 0
total = 500  # Evaluate the first 500 questions

for i in range(total):
    sample_question = train_data[i]["question"]
    correct_answer = train_data[i]["answer"]
    gpt_answer = solve_math_problem(sample_question)  # GPT-4o generated answer

    if check_answer(gpt_answer, correct_answer):
        correct_count += 1
    else:
        print(f"❌ Question {i+1} incorrect")
        print(f"   ✅ Correct answer: {extract_numeric_answer(correct_answer)}")
        print(f"   ❌ GPT answer: {gpt_answer}\n")

# Output accuracy
accuracy = (correct_count / total) * 100
print(f"✅ GPT-4o math solving accuracy: {accuracy:.2f}%")
