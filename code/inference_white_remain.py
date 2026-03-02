from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import jsonlines
import generate_template
import json
import argparse
import torch
torch.cuda.empty_cache()
import inference_utils
from datasets import load_dataset
from huggingface_hub import login

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
    if model_name == "mistral":
        model_ckpt = "mistralai/Mistral-7B-Instruct-v0.2"
    elif model_name == "llama":
        model_ckpt = "meta-llama/Llama-3.2-3B-Instruct"
    elif model_name == "llama2":
        model_ckpt = "meta-llama/Llama-2-13b-chat-hf"
    elif model_name == "qwen2-0.5B":
        model_ckpt = "Qwen/Qwen2-0.5B-Instruct"
    elif model_name == "qwen2-1.5B":
        model_ckpt = "Qwen/Qwen2-1.5B-Instruct"
    elif model_name == "qwen2-7B":
        model_ckpt = "Qwen/Qwen2-7B-Instruct"
   
    
    data = []
    if task == "piqa":
        if data_type == "train":
            question_path = "./dataset/piqa/train.jsonl"
        else:
            question_path = "./dataset/piqa/dev.jsonl"
        # input
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        if (start == -1) or (end == -1):
            print("full!!!")
        else:
            print(f'from {start} to {end}!!')
            data = data[start:]
            print(f'# of data:{len(data)}')
    elif task == "csqa":
        if data_type == "train":
            question_path = "./dataset/csqa/train_rand_split.jsonl"
        else:
            question_path = './dataset/csqa/dev_rand_split.jsonl'
        with jsonlines.open(question_path) as f:
            for line in f.iter():
                data.append(line)
        if (start == -1) or (end == -1):
            print("full!!!")
        else:
            print(f'from {start} to {end}!!')
            data = data[start:]
            print(f'# of data:{len(data)}')
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
        if (start == -1) or (end == -1):
            print("full!!!")
        else:
            print(f'from {start} to {end}!!')
            data = data[start:]
            print(f'# of data:{len(data)}')
    elif task == "arc-h":
        if data_type == "train":
            data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
        else: # test
            data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
    elif task == "obqa":
        if data_type == "train":
            data = load_dataset("allenai/openbookqa", name = "main", split="train")
        else: # test
            data = load_dataset("allenai/openbookqa", name = "main", split="test")
    elif task == "siqa":
        if data_type == "train":
            if (start == -1) or (end == -1):
                print("full!!!")
                data = load_dataset("allenai/social_i_qa", split="train")
            else:
                print(f'from {start} to {end}!!')
                data = load_dataset("allenai/social_i_qa", split=f"train[{start}:]")
        else:
            data = load_dataset("allenai/social_i_qa", split="validation")
    elif task == "wngr":
        if data_type == "train":
            data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")
        else:
            data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
    elif task == "boolq":
        if data_type == "train":
            data = load_dataset("google/boolq", split="train")
        else: # validation
            data = load_dataset("google/boolq", split="validation")
    elif task == "gsm8k":
        if data_type == "train":
            if (start == -1) or (end == -1):
                print("full!!!")
                data = load_dataset("openai/gsm8k", name="main", split="train")
            else:
                print(f'from {start} to {end}!!')
                data = load_dataset("openai/gsm8k", name="main", split=f"train[{start}:]")     
        else: # validation
            data = load_dataset("openai/gsm8k", name="main", split="test")
    elif task == "aqua":
        if data_type == "train":
            if (start == -1) or (end == -1):
                print("full!!!")
                data = load_dataset("deepmind/aqua_rat", name = "raw", split="train")
            else:
                print(f'from {start} to {end}!!')
                data = load_dataset("deepmind/aqua_rat", name = "raw", split=f"train[{start}:{end}]")
        else: # validation
            data = load_dataset("deepmind/aqua_rat", name = "raw", split="test")
    ### France ###
    elif task == "mcsqa":
        if data_type == "train":
            if (start == -1) or (end == -1):
                print("full!!!")
                data = load_dataset("yusuke1997/mCSQA","fr", split="train")
            else:
                print(f'from {start} to {end}!!')
                data = load_dataset("yusuke1997/mCSQA","ru", split=f"train[{start}:]")
            
        else: # validation
            data = load_dataset("yusuke1997/mCSQA","fr", split="test")
    
    
    ### evaluate ###
    outputs = []
    
    # LLMs
    if model_name == "llama2":
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
            max_new_tokens=310,
        )
    else:
        pipe = pipeline(
            "text-generation",
            model=model_ckpt, 
            torch_dtype=torch.bfloat16, 
            max_new_tokens=512, # generative tasks
            device_map="auto",
            do_sample=False
        )

    count = 0
    for instance in data:
        print(f"current idx: {count}")
        if task == "piqa":
            question = instance["goal"]
            sol1 = instance["sol1"]
            sol2 = instance["sol2"]
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.piqa_user_mistral.format(goal=question, sol1=sol1, sol2=sol2)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.piqa_user.format(goal=question, sol1=sol1, sol2=sol2)}
                    ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.piqa_user_2.format(goal=question, sol1=sol1, sol2=sol2)}
                    ]
            
        elif task == "stqa":
            question = instance['question']
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.stqa_user_mistral.format(goal=question)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.stqa_user.format(goal=question)}
                    ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.stqa_user_2.format(goal=question)}
                    ]
            
        elif task == "csqa":
            question = instance['question']['stem']
            sol1 = instance["question"]["choices"][0]['text']
            sol2 = instance["question"]["choices"][1]['text']
            sol3 = instance["question"]["choices"][2]['text']
            sol4 = instance["question"]["choices"][3]['text']
            sol5 = instance["question"]["choices"][4]['text']
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.csqa_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.csqa_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.csqa_user_2.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
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
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.qasc_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.qasc_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8)}
                ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.qasc_user_2.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5, sol6=sol6, sol7=sol7, sol8=sol8)}
                ]

        elif task in ["arc-e", "arc-h"]:
            question = instance['question']
            sol1=instance["choices"]['text'][0] if len(instance["choices"]["text"]) > 0 else ""
            sol2=instance["choices"]['text'][1] if len(instance["choices"]["text"]) > 1 else ""
            sol3=instance["choices"]['text'][2] if len(instance["choices"]["text"]) > 2 else ""
            sol4=instance["choices"]['text'][3] if len(instance["choices"]["text"]) > 3 else ""
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.arc_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.arc_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.arc_user_2.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                ]

        elif task == "obqa":
            question = instance['question_stem']
            sol1=instance["choices"]['text'][0] if len(instance["choices"]["text"]) > 0 else ""
            sol2=instance["choices"]['text'][1] if len(instance["choices"]["text"]) > 1 else ""
            sol3=instance["choices"]['text'][2] if len(instance["choices"]["text"]) > 2 else ""
            sol4=instance["choices"]['text'][3] if len(instance["choices"]["text"]) > 3 else ""
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.obqa_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.obqa_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.obqa_user_2.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4)}
                ]
            
        elif task == "siqa":
            question = instance['question']
            sol1=instance["answerA"]
            sol2=instance["answerB"]
            sol3=instance["answerC"]
            context = instance["context"]
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.siqa_user_mistral.format(context=context, goal=question, sol1=sol1, sol2=sol2, sol3=sol3)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.siqa_user.format(context=context, goal=question, sol1=sol1, sol2=sol2, sol3=sol3)}
                ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.siqa_user_2.format(context=context, goal=question, sol1=sol1, sol2=sol2, sol3=sol3)}
                ]
            
        elif task == "wngr":
            question = instance['sentence']
            sol1=instance["option1"]
            sol2=instance["option2"]
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.wngr_user_mistral.format(goal=question, sol1=sol1, sol2=sol2)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.wngr_user.format(goal=question, sol1=sol1, sol2=sol2)}
                ]
        elif task == "boolq":
            question = instance['question']
            context = instance["passage"]
            if model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.boolq_user.format(context=context, goal=question)}
                ]
                
        # arithmetric tasks
        elif task == "gsm8k":
            question = instance['question']
            if model_name == "llama":
                messages = [
                    {"role": "system", "content": generate_template.system_prompt_arith},
                    {"role": "user", "content": generate_template.gsm8k_user.format(goal=question)}
                    ]
        elif task == "aqua":
            question = instance['question']
            sol1 = instance["options"][0]
            sol2 = instance["options"][1]
            sol3 = instance["options"][2]
            sol4 = instance["options"][3]
            sol5 = instance["options"][4]
            if model_name == "llama":
                messages = [
                    {"role": "system", "content": generate_template.system_prompt_arith},
                    {"role": "user", "content": generate_template.aqua_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]

        elif task == "mcsqa":
            question = instance['question']
            sol1 = instance["choices"]['text'][0]
            sol2 = instance["choices"]['text'][1]
            sol3 = instance["choices"]['text'][2]
            sol4 = instance["choices"]['text'][3]
            sol5 = instance["choices"]['text'][4]
            if model_name == "mistral":
                messages = [
                {"role": "user", "content": generate_template.csqa_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
            elif model_name == "llama":
                messages = [
                {"role": "system", "content": generate_template.system_prompt},
                {"role": "user", "content": generate_template.csqa_user.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
            elif model_name == "llama2":
                messages = [
                {"role": "system", "content": generate_template.system_prompt_2},
                {"role": "user", "content": generate_template.csqa_user_2.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
            elif model_name in ["qwen2-1.5B", "qwen2-7B"]:
                messages = [
                {"role": "system", "content": "You are a helpful assistant.",
                "role": "user", "content": generate_template.csqa_user_mistral.format(goal=question, sol1=sol1, sol2=sol2, sol3=sol3, sol4=sol4, sol5=sol5)}
                    ]
        
        
        
        output = pipe(messages)[0]["generated_text"][-1]['content']

        output = inference_utils.parse_knowledge(output)
        print(output)
        
        outputs.append(output)

        count += 1
    
    if (start == -1) or (end == -1):
        with open(f"./dataset/{task}/converted_{data_type}_{model_name}.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)   
    else:
        with open(f"./dataset/{task}/converted_{data_type}_{model_name}_{start}_{end}.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=2)   