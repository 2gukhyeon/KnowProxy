from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import SYSTEM_HEAD, SYSTEM_CONTENT, USER_HEAD, USER_CONTENT, ASISTANT_HEAD
import jsonlines
import generate_template
import json
import argparse
import torch
torch.cuda.empty_cache()
from datasets import load_dataset
from huggingface_hub import login
import inference_utils

if __name__ == "__main__":
    #### argument #### 
    parser = argparse.ArgumentParser()
    # required = True
    parser.add_argument('--task', help='piqa, csqa, stqa, qasc', type=str, required=True)
    # required = False
    parser.add_argument('--type', help='train or dev', type=str, required=False, default="dev")
    parser.add_argument('--start', help='start idx', type=int, required=False, default=-1)
    parser.add_argument('--end', help='end idx', type=int, required=False, default=-1)
    parser.add_argument('--model_name', help='mistral, llama', type=str, required=False, default="llama")
    parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False, default="your huggingface key")
    args = parser.parse_args()
    
    api_key = args.huggingface_api_key
    login(token=api_key)
    
    # for aqua's train dataset
    start = args.start
    end = args.end
    print(start)
    print(end)
    data_type = args.type
    model_name = args.model_name
    task = args.task
    if model_name == "llama":
        model_ckpt = "meta-llama/Llama-3.2-3B-Instruct"

    
   
    # load datase t
    data = []
    if task == "stqa":
        answer_choices = list("AB")
        if data_type == "train":
            question_path = "./dataset/stqa/train.json"
        else:
            question_path = "./dataset/stqa/dev.json"
        # input
        with open(question_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif task == "qasc":
        answer_choices = list("ABCDEFGH")
        if data_type == "train":
            question_path = "./dataset/qasc/train.jsonl"
        else:
            question_path = "./dataset/qasc/dev.jsonl"
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        if (start == -1) or (end == -1):
            print("full!!!")
        else:
            print(f'from {start} to {end}!!')
            data = data[start:end]
            print(f'# of data:{len(data)}')
    elif task == "obqa":
        answer_choices = list("ABCD")
        if data_type == "train":
            if (start == -1) or (end == -1):
                print("full!!!")
                data = load_dataset("allenai/openbookqa", name = "main", split="train")
            else:
                print(f'from {start} to {end}!!')
                data = load_dataset("allenai/openbookqa", name = "main", split=f"train[{start}:{end}]")
        else: # test
            data = load_dataset("allenai/openbookqa", name = "main", split="test")

    
    # load model
    tokenizer  = AutoTokenizer.from_pretrained(model_ckpt, trust_remote_code=True)
    model      = AutoModelForCausalLM.from_pretrained(
                    model_ckpt,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,)
    model.eval()
    
    outputs = []
    confidence = []

    choice_tokens = {}
    for c in answer_choices:
        choice_tokens[c] = [tokenizer.convert_tokens_to_ids(t) for t in tokenizer.tokenize(f" {c}")]
    
    
    print(choice_tokens)
    outputs, confidences = [], []
    
    for idx, instance in enumerate(data):
        if task == "stqa":
            question = instance['question']
            messages = [
            {"role": "system", "content": generate_template.system_prompt},
            {"role": "user", "content": generate_template.stqa_user_confidence.format(goal=question)}
                ]

 
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
           
            messages = [
            {"role": "system", "content": generate_template.system_prompt},
            {"role": "user", "content": generate_template.qasc_user_confidence.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8)}
            ]
        
        elif task == "obqa":
            question = instance['question_stem']
            sol1=instance["choices"]['text'][0] if len(instance["choices"]["text"]) > 0 else ""
            sol2=instance["choices"]['text'][1] if len(instance["choices"]["text"]) > 1 else ""
            sol3=instance["choices"]['text'][2] if len(instance["choices"]["text"]) > 2 else ""
            sol4=instance["choices"]['text'][3] if len(instance["choices"]["text"]) > 3 else ""

            messages = [
            {"role": "system", "content": generate_template.system_prompt},
            {"role": "user", "content": generate_template.obqa_user_confidence.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
            ]

        prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen_out = model.generate(
                    **inputs,
                    max_new_tokens=256,    
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True)
        
        new_tokens = gen_out.sequences[0][inputs.input_ids.size(-1):]
        
        answer_txt = tokenizer.decode(new_tokens, skip_special_tokens=True)
      
        output = inference_utils.parse_knowledge(answer_txt)
        

        logits_last = gen_out.scores[-2][0]          # shape: (vocab_size,)
        probs_full  = torch.softmax(logits_last, dim=-1)

        for choice in answer_choices:
            if task == "stqa":
                p_A = sum(probs_full[id_] for id_ in choice_tokens["A"])
                p_B = sum(probs_full[id_] for id_ in choice_tokens["B"])
                norm_A = p_A / (p_A + p_B)
                norm_B = p_B / (p_A + p_B)
                if output["Final_Answer"] == "A":
                    final_confidence = norm_A
                else: # B
                    final_confidence = norm_B
            elif task == "obqa":
                p_A = sum(probs_full[id_] for id_ in choice_tokens["A"])
                p_B = sum(probs_full[id_] for id_ in choice_tokens["B"])
                p_C = sum(probs_full[id_] for id_ in choice_tokens["C"])
                p_D = sum(probs_full[id_] for id_ in choice_tokens["D"])
                norm_A = p_A / (p_A + p_B + p_C + p_D)
                norm_B = p_B / (p_A + p_B + p_C + p_D)
                norm_C = p_C / (p_A + p_B + p_C + p_D)
                norm_D = p_D / (p_A + p_B + p_C + p_D)
                if output["Final_Answer"] == "A":
                    final_confidence = norm_A
                elif output["Final_Answer"] == "B":
                    final_confidence = norm_B
                elif output["Final_Answer"] == "C":
                    final_confidence = norm_C
                elif output["Final_Answer"] == "D":
                    final_confidence = norm_D
            elif task == "qasc":
                p_A = sum(probs_full[id_] for id_ in choice_tokens["A"])
                p_B = sum(probs_full[id_] for id_ in choice_tokens["B"])
                p_C = sum(probs_full[id_] for id_ in choice_tokens["C"])
                p_D = sum(probs_full[id_] for id_ in choice_tokens["D"])
                p_E = sum(probs_full[id_] for id_ in choice_tokens["E"])
                p_F = sum(probs_full[id_] for id_ in choice_tokens["F"])
                p_G = sum(probs_full[id_] for id_ in choice_tokens["G"])
                p_H = sum(probs_full[id_] for id_ in choice_tokens["H"])
                norm_A = p_A / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_B = p_B / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_C = p_C / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_D = p_D / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_E = p_E / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_F = p_F / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_G = p_G / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                norm_H = p_H / (p_A + p_B + p_C + p_D + p_E + p_F + p_G + p_H)
                if output["Final_Answer"] == "A":
                    final_confidence = norm_A
                elif output["Final_Answer"] == "B":
                    final_confidence = norm_B
                elif output["Final_Answer"] == "C":
                    final_confidence = norm_C
                elif output["Final_Answer"] == "D":
                    final_confidence = norm_D
                elif output["Final_Answer"] == "E":
                    final_confidence = norm_E
                elif output["Final_Answer"] == "F":
                    final_confidence = norm_F
                elif output["Final_Answer"] == "G":
                    final_confidence = norm_G
                elif output["Final_Answer"] == "H":
                    final_confidence = norm_H
        
        output["Final_Confidence"] = (final_confidence.item())*100
        outputs.append(output)
        
        print(f"[{idx}] answer={output}")
    
    
    ## 저장 코드 수정.

    with open(f"./dataset/{task}/confidence_based_{data_type}_{model_name}_{start}_{end}.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)   