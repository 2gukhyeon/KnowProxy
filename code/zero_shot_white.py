from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from utils import extract_answer
import jsonlines
import json
import argparse
import torch
torch.cuda.empty_cache()
import numpy as np
from datasets import load_dataset
import pandas as pd

if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
    # required = False
    parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    model_ckpt = args.model_ckpt
    model_name = model_ckpt.split("/")[1]
    task = args.task
    
    data = []
    if task == "piqa":
        question_path = "./dataset/piqa/dev.jsonl"
        answer_path = "./dataset/piqa/dev-labels.lst"
        # input
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        # label
        with open(answer_path, 'r') as file:
            answers = [int(line) for line in file.readlines()]
    elif task == "csqa":
        question_path = './dataset/csqa/dev_rand_split.jsonl'
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in data]
    elif task == "stqa":
        question_path = "./dataset/stqa/dev.json"
        # input
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        answers = [0 if line["answer"] else 1 for line in data]
    elif task == "truthfulqa":
        question_path = "./dataset/TruthfulQA/data/v1/mc_task_test.json"
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        labels = "0123456789"
        answers = []
        for line in data:
            targets = line["mc1_targets"]
            for i, (key, val) in enumerate(targets.items()):
                if val == 1:
                    answers.append(int(labels[i]))
    elif task == "scienceqa":
        question_path = "./dataset/scienceqa/scienceqa_test.json"
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        answers = [int(line["answer"]) for line in data] 
    elif task == "qasc":
        question_path = "./dataset/qasc/dev.jsonl"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                    if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                    if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                    if line['answerKey'] == "G" else 7 for line in data]
    elif task == "arc-e":
        data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="test")
        answers = [0 if answer == "A" else 1 if answer == "B" else 2 if answer == "C" else 3 for answer in data["answerKey"]]
    elif task == "arc-h":
        arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3} # 정답값이 이상하게 배치되어 있어서.
        data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
        answers = [arc_label_map[str(answer)] for answer in data["answerKey"]]
    elif task == "obqa":
        data = load_dataset("allenai/openbookqa", name = "main", split="test")
        answers = [0 if answer == "A" else 1 if answer == "B" else 2 if answer == "C" else 3 for answer in data["answerKey"]]
    elif task == "siqa":
        data = load_dataset("allenai/social_i_qa", split="validation")
        answers = data["label"] # 1,2,3
        answers = [int(answer)-1 for answer in answers] # 0,1,2 로 changing!! 
    elif task == "wngr":
        data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
        answers = data["answer"]
        answers = [int(answer)-1 for answer in answers]   
    elif task == "boolq":
        data = load_dataset("google/boolq", split="validation")
        answers = data["answer"]
        answers = [0 if answer else 1 for answer in answers]
    elif task == "gsm8k":
        data = load_dataset("openai/gsm8k", name = "main", split="test")
        answers = data["answer"]
        answers = [extract_answer(task, label) for label in answers]
    elif task == "aqua":
        data = load_dataset("deepmind/aqua_rat", name = "raw", split="test")
        answers = data["correct"]
        answers = [extract_answer(task, label) for label in answers]
    elif task == "ko-arc":
        arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3} # 정답값이 이상하게
        data = load_dataset("davidkim205/ko_arc_challenge", split="test")
        answers = [arc_label_map[str(answer)] for answer in data["answerKey"]]
    elif task == "mcsqa":
        data = load_dataset("yusuke1997/mCSQA","fr", split=f"test")
        answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in data]
    
    
    ### evaluate ###
    outputs = []
    
    if model_name == "Llama-2-13b-chat-hf": # Llama 2
        tokenizer = AutoTokenizer.from_pretrained(
            model_ckpt,
            use_fast=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_ckpt,
            load_in_4bit=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            do_sample=False, 
            max_new_tokens=64,
        )
        print("run Llama 2!")
    else: # Llama 3.2, Mistral
        pipe = pipeline(
            "text-generation", 
            model=model_ckpt, 
            torch_dtype=torch.bfloat16, 
            max_new_tokens=512, # gsm8k, aqua; it is generative task. 
            # max_new_tokens=16, # other
            device_map="auto",
            do_sample=False
        )
        print(f"run {model_name}!")
    
    for instance in data:
        if task == "piqa":
            prompt = f"Question: {instance['goal']}\nChoices:\nA. {instance['sol1']}\nB. {instance['sol2']}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # PIQA
        elif task == "stqa":
            prompt = f"Question: {instance['question']}\nChoices:\nA. True\nB. False\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # QASC
        elif task == "truthfulqa":
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            def format_question(question, targets):
                options = []
                for i, key in enumerate(targets.keys()):
                    options.append(f"{labels[i]}. {key}")
                return f"Question: {question}\nChoices:\n" + "\n".join(options) + "\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("
            
            prompt = format_question(instance["question"], instance["mc1_targets"])
        elif task == "scienceqa":
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            def format_question(question, targets):
                options = []
                for i, key in enumerate(targets):
                    options.append(f"{labels[i]}. {key}")
                return f"Question: {question}\nChoices:\n" + "\n".join(options) + "\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is ("
            
            prompt = format_question(instance["question"], instance["choices"])  
        elif task == "csqa":
            question = instance['question']['stem']
            sol1 = instance["question"]["choices"][0]['text']
            sol2 = instance["question"]["choices"][1]['text']
            sol3 = instance["question"]["choices"][2]['text']
            sol4 = instance["question"]["choices"][3]['text']
            sol5 = instance["question"]["choices"][4]['text']
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # CSQA
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
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nF. {sol6}\nG. {sol7}\nH. {sol8}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # QASC
        elif task in ["arc-e", "arc-h", "ko-arc"]:
            question = instance['question']
            sol1=instance["choices"]['text'][0] if len(instance["choices"]["text"]) > 0 else ""
            sol2=instance["choices"]['text'][1] if len(instance["choices"]["text"]) > 1 else ""
            sol3=instance["choices"]['text'][2] if len(instance["choices"]["text"]) > 2 else ""
            sol4=instance["choices"]['text'][3] if len(instance["choices"]["text"]) > 3 else ""
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # ARC
        elif task == "obqa":
            question = instance['question_stem']
            sol1=instance["choices"]['text'][0] if len(instance["choices"]["text"]) > 0 else ""
            sol2=instance["choices"]['text'][1] if len(instance["choices"]["text"]) > 1 else ""
            sol3=instance["choices"]['text'][2] if len(instance["choices"]["text"]) > 2 else ""
            sol4=instance["choices"]['text'][3] if len(instance["choices"]["text"]) > 3 else ""
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" 
        elif task == "siqa":
            question = instance['question']
            sol1=instance["answerA"]
            sol2=instance["answerB"]
            sol3=instance["answerC"]
            context=instance["context"] # siqa의 특성.
            prompt = f"Context: {context}\nQuestion: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # ARC
        
        elif task == "wngr":
            question = instance['sentence']
            sol1=instance["option1"]
            sol2=instance["option2"]
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # ARC
        elif task == "boolq":
            question = instance['question']
            context=instance["passage"] # ref. boolq paper
            prompt = f"Context: {context}\nQuestion: {question}\nChoices:\nA. True\nB. False\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # ARC
        # generation
        elif task == "gsm8k":
            question = instance["question"]
            prompt = f"Question: {question}\nAssistant: Therefore, the answer (arabic numerals) is" # ref. Zero-shot CoT's zero-shot code.
        elif task == "aqua":
            question = instance["question"]
            sol1=instance["options"][0]
            sol2=instance["options"][1]
            sol3=instance["options"][2]
            sol4=instance["options"][3]
            sol5=instance["options"][4]
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nAssistant: Therefore, among A through E, the answer is"
        elif task == "mcsqa":
            question = instance['question']
            sol1 = instance["choices"]['text'][0]
            sol2 = instance["choices"]['text'][1]
            sol3 = instance["choices"]['text'][2]
            sol4 = instance["choices"]['text'][3]
            sol5 = instance["choices"]['text'][4] 
            prompt = f"Question: {question}\nChoices:\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5}\nHuman: Given all of the above, what's the single, most likely answer?\nAssistant: The single, most likely answer is (" # CSQA
        if task in ["gsm8k", "scienceqa"]: # free-generative tasks
            messages = [
                # {"role": "system", "content": "You are a helpful AI assistant."}, # remove
                {"role": "user", "content": prompt}
            ]
        else: # MC tasks
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Given a question, choose the single, most likely answer."},
                {"role": "user", "content": prompt}
            ]
        output = pipe(messages)[0]["generated_text"][-1]['content']
        # print(output)
        if model_name == "Llama-2-13b-chat-hf": # Llama 2
            try:
                output = output.split("\n\n")[1] # extracting only answers
            except:
                output = output # pass
        print(output)
        if ("I'm so sorry" in output) or ("I apologize" in output) or ("As a helpful AI assistant" in output): # for llama2 
            outputs.append(-1)
            print("avoiding response, so it is error!")
            continue
    
        if task in ["gsm8k", "aqua"]:
            outputs.append(extract_answer(task, output))
        else:            
            if output.strip().strip(".") == "True":
                outputs.append(0)
            elif output.strip().strip(".") == "False":
                outputs.append(1)
            else:
                for pred in output:
                    if pred == "A":
                        outputs.append(0)
                        break
                    elif pred == "B":
                        outputs.append(1)
                        break
                    elif pred == "C":
                        outputs.append(2)
                        break
                    elif pred == "D":
                        outputs.append(3)
                        break
                    elif pred == "E":
                        outputs.append(4)
                        break
                    elif pred == "F":
                        outputs.append(5)
                        break
                    elif pred == "G":
                        outputs.append(6)
                        break
                    elif pred == "H":
                        outputs.append(7)
                        break
                else:
                    outputs.append(-1)
                    
    error_case = []
    for idx in range(len(outputs)):
        if outputs[idx] != answers[idx]:
            error_case.append(idx)
    
    acc = np.sum(np.where(np.array(outputs) == np.array(answers), 1, 0)) *100 / len(answers)
    print(f'{task} {model_name} zero-shot acc: {acc}')
    # df = pd.DataFrame(error_case, columns=["idx"])
    # df.to_csv(f"./results/llama_confusion_matrix/{task}/error_case/{task}_False_error_case_3b.csv")
    
    

