import re
import json
import csv
import jsonlines
import PyPDF2
import google.generativeai as genai

#Gemini setting
GOOGLE_API_KEY = ''
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

#read PDF
with open('Optimisation1.pdf', mode='rb') as mypdf:
    reader = PyPDF2.PdfReader(mypdf)
    all_text = "\n".join(page.extract_text() for page in reader.pages)


#questions
question_blocks = re.split(r'\n(?=\d+\.\s*\[\d+\s*marks\s*\])', all_text)

data = []

for i, block in enumerate(question_blocks):
    block = block.strip()
    if not block:
        continue

    #split question from solution
    parts = re.split(r'\n*Solution\.\n*', block, maxsplit=1, flags=re.IGNORECASE)

    if len(parts) != 2:
        continue  

    full_problem, full_solution = parts
    full_problem = full_problem.strip()
    full_solution = full_solution.strip()

    #Check if it contains subproblems
    subproblem_pattern = r'\([ivxlc]+\)\s*(.*?)\n*Solution\.\n*(.*?)(?=\n\([ivxlc]+\)|$)'
    subproblems = re.findall(subproblem_pattern, block, flags=re.DOTALL)

    #if no subproblems found
    if not subproblems:
        #feed Gemini with problems
        prompt = f"(question to be solved is)\n{full_problem}\n(give me the proof process and final result, combine the process and result in one paragraph with a double dash before the content)"
        try:
            response = model.generate_content(prompt)
            ai_answer = response.text.split('--')[1].strip() if '--' in response.text else response.text.strip()
        except Exception as e:
            ai_answer = f"Error from Gemini: {e}"

        #store question number
        match = re.match(r'(\d+)\.\s*\[\d+\s*marks\s*\]', full_problem)
        question_num = match.group(1) if match else str(i+1)

        data.append({
            "question_number": question_num,
            "problem": full_problem,
            "gemini_answer": ai_answer,
            "true_solution": full_solution
        })
    else:
        #solve each subproblem
        question_num = str(i + 1)

        for subproblem, solution in subproblems:
            #feed Gemini with subproblems
            prompt = f"(question to be solved is)\n{subproblem.strip()}\n(give me the proof process and final result, combine the process and result in one paragraph with a double dash before the content)"
            try:
                response = model.generate_content(prompt)
                ai_answer = response.text.split('--')[1].strip() if '--' in response.text else response.text.strip()
            except Exception as e:
                ai_answer = f"Error from Gemini: {e}"

            #store all problem, answers, and solutions
            data.append({
                "question_number": f"{question_num} ({subproblem[:3]})",  
                "problem": subproblem.strip(),
                "gemini_answer": ai_answer,
                "true_solution": solution.strip()
            })

#save results as jsonl file
with jsonlines.open("gemini_processed.jsonl", mode='w') as writer:
    for item in data:
        writer.write(item)

#save results as csv file
with open("gemini_processed.csv", mode='w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ["question_number", "problem", "gemini_answer", "true_solution"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for item in data:
        writer.writerow(item)

print(f"\n✅ Processed {len(data)} questions and subproblems.")
print("✅ Saved to gemini_processed.jsonl and gemini_processed.csv")
