import numpy as np
import pandas as pd
import json
import jsonlines
import ast
import pathlib
import textwrap
import time

import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# Use Gemini
API_KEY='api'

genai.configure(api_key=API_KEY)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

model = genai.GenerativeModel('gemini-2.0-flash')

# Answer the questions
n = 0
cc = 0
qstart = 0
rpm = 15
task_per = 1319
answer_ai = np.zeros(task_per)
gsm = 'datasets/gsm8k.jsonl'
gap = 979

df_new = 1
# store the results
question_number = []
ai_a = []
correct_a = []
check = []

with open(gsm, 'r', encoding = 'utf8') as q_file:
    for i in range(gap - 1):
        q_file.readline()
        
    for partition in range(gap - 1, task_per, rpm):
        end = min(partition + rpm, task_per) 
        
        for _ in range(partition, end):
            current_entity = q_file.readline()
            if not current_entity:
                break
            entity_dict = ast.literal_eval(current_entity)
            response = model.generate_content(f'{entity_dict['question']},(no process, give me the answer as a plain number, without quotes.)')
            
            answer_ai = response.text
            correct_answer = float(entity_dict['answer'].split('\n#### ')[1].replace(',',''))
            YorN = 1 if float(answer_ai) == correct_answer else 0
            
            # results
            question_number.append(_ + 1)
            ai_a.append(float(answer_ai))
            correct_a.append(correct_answer)
            check.append(YorN)
            
            print('Question', _+1)
            print('Gemini answer : ', float(answer_ai))
            print('Correct answer : ', float(entity_dict['answer'].split('#### ')[1].replace(',','')))
            if YorN == 1 :
                print('✅')
            else:
                print('❌')
        
        print(f'Completed partition {partition + 1} to {end}. Sleeping for 90 seconds...\n')
        time.sleep(90)


# save results in dataFrame
df = pd.DataFrame({
    'Question': question_number,
    'AI answer': ai_a,
    'Correct answer': correct_a,
    'Match': check})
print(df)

print('Gemini Accuracy: {:.2%}'.format(sum(df['Match'])/max(question_number)))

# Save results to csv and txt files
df.to_csv('results.csv', index=False)


# tool
df_new = pd.DataFrame({
    'Question': question_number,
    'AI answer': ai_a,
    'Correct answer': correct_a,
    'Match': check})
print(df_new)

df = pd.concat([df,df_new], ignore_index = True)

df_new = df_new.drop(range(0, 17))

df = df.drop(range(0, 800))

df = df.drop(849)

df.drop(849)

df_new.drop(range(0, 17))

df = df.reset_index()

df = df.drop('index', axis = 1)

df_new = df_new.reset_index()

df_new = df_new.drop('index', axis = 1)

df_new = 1

df_new = pd.DataFrame({
    'Question': 1002,
    'AI answer': 1.2,
    'Correct answer': 2.0,
    'Match': 0}, index = [511])
print(df_new)


csv152 = pd.read_csv('results(152).csv', header = 0)
csv152_df = pd.DataFrame(csv152)
print(csv152_df)





