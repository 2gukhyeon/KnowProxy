import re
import numpy as np
import evaluate   # pip install evaluate
import os
import re


f1_metric = evaluate.load("f1") # f1 score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # micro F1
    micro_f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="micro"
    )["f1"]

    # macro F1
    macro_f1 = f1_metric.compute(
        predictions=preds,
        references=labels,
        average="macro"
    )["f1"]

    return {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }


def duplicate(label):
    # ','나 'or'이 있으면 2개 이상이라고 판단 → -1
    if 'or' in label:
        return True
    # 단일 정답 매칭
    else:
        return False
    
# ref. Plan-and-Solve's extrating code.
def extract_number(text: str):
    text = text.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', text)]
    if pred:
        pred_answer = float(pred[-1])
    else:
        pred_answer = None
    return pred_answer


def proxy_answer(text, task):
    if task == "qasc":
        return int(text) # QASC는 알파벳으로 라벨링하면 "G"로만 출력하는 문제점이 발생한다.
    else:
        for pred in text:
            if pred == "A":
                return "A"
            elif pred == "B":
                return "B"
            elif pred == "C":
                return "C"
            elif pred == "D":
                return "D"
            elif pred == "E":
                return "E"
            elif pred == "F":
                return "F"
            elif pred == "G":
                return "G"
            elif pred == "H":
                return "H"
        else:
            return "Z"

def extract_answer(task, text):
    if task in ["svamp", "gsm8k", "multiarith", "addsub", "singleeq"]:
        pred_answer = extract_number(text)
    elif task == "commonsenseqa":
        pred = text.strip()
        pred = re.sub("\(|\)|\:|\.|\,", "", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ('A|B|C|D|E')][-1]
        # pred_answer = re.findall(r'A|B|C|D|E', pred)[0]
        return pred_answer
    elif task in ["aqua", "truthfulqa", "scienceqa"]:
        pred = text.strip()
        try:
            pred_answer = re.findall(r'A|B|C|D|E', pred)
            pred_answer = pred_answer[-1]
        except:
            pred_answer = "Z" # Error
        return pred_answer
    elif task == "strategyqa" or task == 'coin_flip':
        pred = text.lower()
        pred = re.sub("\"|\'|\n|\.|\s|\:|\,", " ", pred)
        pred = pred.split()
        pred_answer = [i for i in pred if i in ("yes", "no")][-1]
        return pred_answer
    elif task == "last_letters":
        pred = re.sub("\"|\'|\n|\.|\s", "", text)
        pred_answer = pred
        return pred_answer
    else:
        raise NotImplementedError(' not support task: {}'.format(task))
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def target_module(model):
    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)

    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))

    return target_modules

def answer_prompt(task):
    # ref. Zero-shot CoT code.
    if task == "piqa":
        direct_answer_trigger = "Therefore, among A through B, the answer is"
    elif task == "csqa":
        direct_answer_trigger = "Therefore, among A through E, the answer is"
    elif task == "qasc":
        direct_answer_trigger = "Therefore, among A through H, the answer is"
    elif task in ["stqa", "boolq"]:
        direct_answer_trigger = "Therefore, among A through B, the answer is"
    elif task == "siqa":
        direct_answer_trigger = "Therefore, among A through C, the answer is"
    elif task == "wngr":
        direct_answer_trigger = "Therefore, among A through B, the answer is"
    elif task == "arc-h":
        direct_answer_trigger = "Therefore, among A through D, the answer is"
    elif task == "obqa":
        direct_answer_trigger = "Therefore, among A through D, the answer is"
    elif task in ["gsm8k", "svamp"]:
        direct_answer_trigger = "Therefore, the answer (arabic numerals) is"
    elif task == "aqua":
        direct_answer_trigger = "Therefore, among A through E, the answer is"   
    elif task == "truthfulqa":
        direct_answer_trigger = "Therefore, the answer is"   
    # direct_answer_trigger = "Therefore,the answer is" # ref. Plan-and-Solve code.
    return direct_answer_trigger

def detecting_ckpt(base_dir):
    ckpt = []
    for folder in os.listdir(base_dir):
        match = re.search(r"(\d+)$", folder)  # ckpt 
        if match:
            ckpt.append(int(match.group(1)))

    ckpt = sorted(ckpt)
    return ckpt

# CoT prompt (Baselines)
ps_plus_rea_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, and devise a plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numeral calculation and commonsense), solve the problem step by step, and show the answer."
ps_rea_prompt = "Let's first understand the problem and devise a plan to solve the problem. Then, let's carry out the plan to solve the problem step by step."
cot_rea_prompt = "Let's think step by step."

# llama 3.2 prompt template
SYSTEM_HEAD = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
SYSTEM_CONTENT = "{system_prompt}\n\n"
USER_HEAD = "<|start_header_id|>user<|end_header_id|>\n"
USER_CONTENT = "{user}"
ASISTANT_HEAD = "<|start_header_id|>assistant<|end_header_id|>\n"
