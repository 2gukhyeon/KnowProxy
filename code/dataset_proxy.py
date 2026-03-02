import jsonlines
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
from utils import duplicate

# commonsense reasoning tasks
piqa_instruction = '[INST] {question}\nA. {sol1}\nB. {sol2} [/INST] The answer is:'
csqa_instruction = '[INST] {question}\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4}\nE. {sol5} [/INST] The answer is:'
qasc_instruction = '[INST] {question}\n1. {sol1}\n2. {sol2}\n3. {sol3}\n4. {sol4}\n5. {sol5}\n6. {sol6}\n7. {sol7}\n8. {sol8} [/INST] The answer is:'
# qasc_instruction = '[INST] {question}\nA. {sol1} B. {sol2} C. {sol3} D. {sol4} E. {sol5} F. {sol6} G. {sol7} H. {sol8} [/INST] The answer is:'
arc_instruction = '[INST] {question}\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4} [/INST] The answer is:'
obqa_instruction = '[INST] {question}\nA. {sol1}\nB. {sol2}\nC. {sol3}\nD. {sol4} [/INST] The answer is:'
siqa_instruction = '[INST] {context}\n{question}\nA. {sol1}\nB. {sol2}\nC. {sol3} [/INST] The answer is:'
wngr_instruction = '[INST] {question}\nA. {sol1}\nB. {sol2} [/INST] The answer is:'
boolq_instruction = '[INST] {context}\n{question}\nA. True\nB. False [/INST] The answer is:'
stqa_instruction = '[INST] {question}\nA. True\nB. False [/INST] The answer is:'

IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# For cls
class ProxyDataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True): 
        self.q_path = q_path
        self.l_path = l_path
        self.data = []
        self.tokenizer = tokenizer

        self.task = task
        self.is_train = is_train
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # token 추가하지 않음. token 일정하게 유지.
        
        # load question
        if self.task == "piqa":
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)

            # load label
            with open(self.l_path, 'r') as file:
                self.answers = [line.strip() for line in file.readlines()]
            
            self.answers = ["A" if int(answer) == 0 else "B" for answer in self.answers]
            
        elif self.task == "csqa":
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
            
            self.answers = [line["answerKey"].strip() for line in self.data]
    
        elif self.task == "qasc":
            qasc_label_map = {'A': "1", 'B': "2", 'C': "3", 'D': "4",'E': "5", 'F': "6", 'G': "7", 'H': "8"}
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
            self.answers = [line["answerKey"].strip() for line in self.data]
            self.answers = [int(qasc_label_map[str(answer)]) for answer in self.answers]
       
        elif self.task == "arc-h":
            arc_label_map = {'A': "A", 'B': "B", 'C': "C", 'D': "D",
                             '1': "A", '2': "B", '3': "C", '4': "D"}
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
            else: # validation
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
 
            answers = self.data["answerKey"]
            self.answers = [arc_label_map[str(answer)] for answer in answers]
            print(f'first # of answer: {len(self.answers)}')
        elif self.task == "obqa":
            if self.is_train:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="train")
            else:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="test")

            answers = self.data["answerKey"]
            self.answers = [answer for answer in answers]
            
        elif self.task == "siqa":
            if self.is_train:
                self.data = load_dataset("allenai/social_i_qa", split="train")
            else:
                self.data = load_dataset("allenai/social_i_qa", split="validation")
       
            answers = self.data["label"]
        
            self.answers = ["A" if int(answer) == 1 else "B" if int(answer) == 2 else "C" for answer in answers]
            
        elif self.task == "wngr":
            if self.is_train:
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")   
            else: # validation
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
            answers = self.data["answer"]
           
            self.answers = ["A" if int(answer) == 1 else "B" for answer in answers]
        elif self.task == "stqa":
            with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            self.answers = ["A" if line["answer"] else "B" for line in self.data]
            
        elif self.task == "boolq":
            if self.is_train:
                self.data = load_dataset("google/boolq", split="train") 
            else: # validation
                self.data = load_dataset("google/boolq", split="validation")
            self.answers = ["A" if line["answer"] else "B" for line in self.data]
        
        
        if is_train:
            self.prepare_sft_dataset()
        else:
            self.validation()

    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            add_special_tokens=True,
            padding=True,
            truncation="longest_first",
            max_length=256
        )
        
        return encoding
    
    def huggingface_formatting(self):
        """
        에러가 존재하여, ARC challgenge는 별도의 함수를 구축하여, 처리한다.
        """
        idx = 0
        prompts = []
        final_answer = []
        for item in self.data:
            choices = item['choices']['text']
            if self.task in ["arc-e", "arc-h"]:
                question = item['question']
            elif self.task == "obqa":
                question = item['question_stem']
            
            if self.task in ["arc-e", "arc-h"]:
                template = arc_instruction
            elif self.task == "obqa":
                template = obqa_instruction
                
            if len(choices) == 4:
                formatted_prompt = template.format(
                    question=question,
                    sol1=choices[0],
                    sol2=choices[1],
                    sol3=choices[2],
                    sol4=choices[3]
                )
                final_answer.append(self.answers[idx])
                prompts.append(formatted_prompt)
                idx += 1
            else:
                idx += 1
                continue
        self.answers = final_answer
        return prompts
    
    def prepare_sft_dataset(self, verbose=True):
        if self.task == "piqa":
            prompts = [piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]) for item in self.data]
        elif self.task == "csqa":
            prompts = [csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']) for item in self.data]
        elif self.task == "qasc":
            prompts = [qasc_instruction.format(question=item['question']['stem'].strip(), sol1=item['question']['choices'][0]['text'].strip(), sol2=item['question']['choices'][1]['text'].strip(), sol3=item['question']['choices'][2]['text'].strip(), sol4=item['question']['choices'][3]['text'].strip(),
                                                              sol5=item['question']['choices'][4]['text'].strip(), sol6=item['question']['choices'][5]['text'].strip(), sol7=item['question']['choices'][6]['text'].strip(), sol8=item['question']['choices'][7]['text'].strip()) for item in self.data]
        elif self.task in ["arc-h", "obqa"]:
            prompts = self.huggingface_formatting()
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            prompts = [siqa_instruction.format(context=item["context"],question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]) for item in self.data]
        elif self.task == "wngr":
            prompts = [wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]) for item in self.data]
        elif self.task == "stqa":
            prompts = [stqa_instruction.format(question=item['question']) for item in self.data]
        elif self.task == "boolq":
            prompts = [boolq_instruction.format(context=item["passage"], question=item['question']) for item in self.data]
        answers = [str(item) for item in self.answers]
        # texts = [str(prompt) + str(answer) + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)]
        texts = [str(prompt) + str(answer) for prompt, answer in zip(prompts, answers)]
        self.encoding = self.tokenizing(texts)
        input_ids = self.encoding["input_ids"]
        self.labels = input_ids.clone()
        

        for idx in range(len(self.labels)):
            self.labels[idx,:len(self.tokenizing(prompts[idx])['input_ids'][0])-1] = IGNORE_INDEX
        
        # torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {texts[0]}')
            
            print(self.labels[0])
            print(self.labels[1])

        return None
    
    def validation(self):
        # inputs
        if self.task == "piqa":
            self.inputs = [piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]) for item in self.data]
        elif self.task == "csqa":
            self.inputs = [csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']) for item in self.data]
        elif self.task == "qasc":
            self.inputs = [qasc_instruction.format(question=item['question']['stem'].strip(), sol1=item['question']['choices'][0]['text'].strip(), sol2=item['question']['choices'][1]['text'].strip(),
                                                    sol3=item['question']['choices'][2]['text'].strip(), sol4=item['question']['choices'][3]['text'].strip(),
                                                    sol5=item['question']['choices'][4]['text'].strip(), sol6=item['question']['choices'][5]['text'].strip(),
                                                    sol7=item['question']['choices'][6]['text'].strip(), sol8=item['question']['choices'][7]['text'].strip()) for item in self.data]
        elif self.task in ["arc-h", "obqa"]:
            self.inputs = self.huggingface_formatting()
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            self.inputs = [siqa_instruction.format(context=item["context"], question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]) for item in self.data]
        elif self.task == "wngr":
            self.inputs = [wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]) for item in self.data]
        elif self.task == "stqa":
            self.inputs = [stqa_instruction.format(question=item['question']) for item in self.data]
        elif self.task == "boolq":
            self.inputs = [boolq_instruction.format(context=item["passage"], question=item['question']) for item in self.data]
        # labels
        self.labels = self.answers
        
        
        # print(self.inputs[0])
        # print(self.labels[0])
        return None
    
    def __getitem__(self, index):
        data = {key: val[index] for key, val in self.encoding.items()}
        data['labels'] = self.labels[index]
        return data 

    def __len__(self):
        return len(self.labels)  
        

if __name__ == "__main__":
    """"
    Instruction model에도 INST라는 토큰은 없다.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    print(tokenizer.convert_ids_to_tokens(70956))
    print("####")
    question_path = "./dataset/piqa/train.jsonl"
    answer_path = "./dataset/piqa/train-labels.lst"
    # tokenizer.padding_side = "right" # standard
    data = ProxyDataset("siqa", question_path, answer_path, tokenizer, True)
    print(data[0])
    
