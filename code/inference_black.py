import jsonlines
import json
from openai import OpenAI
import argparse
import time
import os
import generate_template
import torch
torch.cuda.empty_cache()
from datasets import load_dataset
from inference_utils import parse_knowledge
openai_api_key = os.getenv("OPENAI_API_KEY")  
# API setting constants
API_MAX_RETRY = 5
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"

parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
# required = False
parser.add_argument('--type', help='train or dev', type=str, required=False, default="dev")
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="gpt-3.5-turbo")

args = parser.parse_args()
data_type = args.type
model_ckpt = args.model_ckpt
task = args.task



def chat_completion(system, prompt):
    for _ in range(API_MAX_RETRY):
        try:
            client = OpenAI(api_key=openai_api_key)
            # gpt-3.5-turbo
            # response = client.chat.completions.create(
            #   model=model_ckpt,
            #   messages=[
            #         {"role": "system", "content": system},
            #         {"role": "user", "content": prompt},
            #     ],
            #   temperature=0,
            #   top_p=1,
            # )
            # gpt-5-mini
            response = client.chat.completions.create(
              model=model_ckpt,
              messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            #   temperature=0,
              top_p=1,
            )
            # response = response.choices
            response = response.choices[0].message.content
            # print(response)
            response = parse_knowledge(response)

            return response
        
        except Exception as e:
            if 'policy' in str(e):
                print("Skipping due to openai policy")
                return 'None'
            print(type(e), e)
            print("trying again")
            time.sleep(API_RETRY_SLEEP)

    return "None"


data = []
if task == "piqa":
    if data_type == "train":
        question_path = "./dataset/piqa/train.jsonl"
        answer_path = "./dataset/piqa/train-labels.lst"
    else:
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
    # input
    with jsonlines.open(question_path) as f:
        for line in f.iter():
            data.append(line)

elif task == "csqa":
    if data_type == "train":
        question_path = "./dataset/csqa/train_rand_split.jsonl"
    else:
        question_path = './dataset/csqa/dev_rand_split.jsonl'
    with jsonlines.open(question_path) as f:
        for line in f.iter():
            data.append(line)

elif task == "stqa":
    if data_type == "train":
        question_path = "./dataset/stqa/train.json"
    else:
        question_path = "./dataset/stqa/dev.json"
    # input
    with open(question_path, "r", encoding="utf-8") as f:
        data = json.load(f)

elif task == "qasc":
    if data_type == "train":
        question_path = "./dataset/qasc/train.jsonl"
    else:
        question_path = "./dataset/qasc/dev.jsonl"
    with jsonlines.open(question_path) as f:
        for line in f.iter():
            data.append(line)
elif task == "arc-h":    
    if data_type == "train":
        data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
    else: # test
        data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
elif task == "wngr":
    if data_type == "train":
        data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")
    else: # validation
        data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
elif task == "obqa":
    if data_type == "train":
        data = load_dataset("allenai/openbookqa", name = "main", split="train")
    else: # test
        data = load_dataset("allenai/openbookqa", name = "main", split="test")
elif task == "siqa":
    if data_type == "train":
        data = load_dataset("allenai/social_i_qa", split="train")
    else: # validation
        data = load_dataset("allenai/social_i_qa", split="validation")
elif task == "boolq":
    if data_type == "train":
        data = load_dataset("google/boolq", split="train")
    else: # validation
        data = load_dataset("google/boolq", split="validation")
elif task == "gsm8k":
    if data_type == "train":
        data = load_dataset("openai/gsm8k", name="main", split="train")
    else: # validation
        data = load_dataset("openai/gsm8k", name="main", split="test")
### evaluate ###
outputs = []

count = 0
for instance in data:
    print(f"current idx: {count}")
    if task == "piqa":
        question = instance["goal"]
        sol1 = instance["sol1"]
        sol2 = instance["sol2"]
        output = chat_completion(generate_template.system_gpt_ab, generate_template.piqa_gpt.format(goal=question, sol1=sol1, sol2=sol2))
        
    elif task == "stqa":
        question = instance['question']
        output = chat_completion(generate_template.system_gpt_ab, generate_template.stqa_gpt.format(goal=question))
        
    elif task == "csqa":
        question = instance['question']['stem']
        sol1 = instance["question"]["choices"][0]['text']
        sol2 = instance["question"]["choices"][1]['text']
        sol3 = instance["question"]["choices"][2]['text']
        sol4 = instance["question"]["choices"][3]['text']
        sol5 = instance["question"]["choices"][4]['text']
        output = chat_completion(generate_template.system_gpt_ae, generate_template.csqa_gpt.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5))
    elif task == "qasc":
        question = instance['question']['stem']
        sol1 = instance['question']['choices'][0]['text']
        sol2 = instance["question"]["choices"][1]['text']
        sol3 = instance["question"]["choices"][2]['text']
        sol4 = instance["question"]["choices"][3]['text']
        sol5 = instance["question"]["choices"][4]['text']
        sol6 = instance["question"]["choices"][5]['text']
        sol7 = instance["question"]["choices"][6]['text']
        sol8 = instance["question"]["choices"][7]['text']
        output = chat_completion(generate_template.system_gpt_ah, generate_template.qasc_gpt.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8))
    elif task == "obqa":
        question = instance['question_stem']
        choices = instance['choices']['text']
        sol1 = choices[0] if len(choices) > 0 else ""
        sol2 = choices[1] if len(choices) > 1 else ""
        sol3 = choices[2] if len(choices) > 2 else ""
        sol4 = choices[3] if len(choices) > 3 else ""
        output = chat_completion(generate_template.system_gpt_ad, generate_template.obqa_gpt.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4))
    elif task == "arc-h":
        question = instance['question']
        choices = instance['choices']['text']
        sol1 = choices[0] if len(choices) > 0 else ""
        sol2 = choices[1] if len(choices) > 1 else ""
        sol3 = choices[2] if len(choices) > 2 else ""
        sol4 = choices[3] if len(choices) > 3 else ""
        output = chat_completion(generate_template.system_gpt_ad, generate_template.arc_h_gpt.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4))
    elif task == "wngr":
        question = instance['sentence']
        sol1 = instance["option1"]
        sol2 = instance["option2"]
        output = chat_completion(generate_template.system_gpt_ab, generate_template.wngr_gpt.format(goal=question, sol1=sol1, sol2=sol2))
    elif task == "siqa":
        question = instance['question']
        sol1=instance["answerA"]
        sol2=instance["answerB"]
        sol3=instance["answerC"]
        context = instance["context"]
        output = chat_completion(generate_template.system_gpt_siqa, generate_template.siqa_gpt.format(context=context, goal=question, sol1=sol1, sol2=sol2, sol3=sol3))
    elif task == "boolq":
        question = instance['question']
        context = instance["passage"]
        output = chat_completion(generate_template.system_gpt_ab, generate_template.boolq_gpt.format(context=context, goal=question))
    
    # arithmetric tasks
    elif task == "gsm8k":
        question = instance['question']
        output = chat_completion(generate_template.system_gpt_arith, generate_template.gsm8k_gpt.format(goal=question))
    
    print(output)
    outputs.append(output)

    count += 1
    
with open(f"./dataset/{task}/converted_{data_type}_gpt5_6k.json", "w", encoding="utf-8") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)  