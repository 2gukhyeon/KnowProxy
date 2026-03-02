import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
import random
random.seed(42)

# input format
piqa_instruction = '[physicaliqa]:\n<goal>{question}</goal>\n<sol1>{sol1}</sol1>\n<sol2>{sol2}</sol2>'
csqa_instruction = '[commonsenseqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>'
stqa_instruction = '[strategyqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'
qasc_instruction = '[qasc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>\n<option6>{sol6}</option6>\n<option7>{sol7}</option7>\n<option8>{sol8}</option8>'
arc_instruction = '[arc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
obqa_instruction = '[obqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
siqa_instruction = '[siqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>'
wngr_instruction = '[wngr]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# For cls
class BaselineDataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True, knowledge_path=None, baseline=None): 
        self.q_path = q_path
        self.l_path = l_path
        self.data = []
        self.tokenizer = tokenizer
        self.task = task
        self.is_train = is_train
        self.baseline = baseline
        
        # load question
        if self.task == "piqa":
            if self.baseline == "gkp":
                if self.is_train:
                    with open("./knowledges/gkp_gpt3_knowledge/knowledge_gkp_gpt3curie.train.piqa.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                else:
                    with open("./knowledges/gkp_gpt3_knowledge/gkp/piqa/knowledge_gkp_gpt3curie.dev.piqa.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)

            # raw dataset
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
            questions = [item['goal'] for item in self.data]
            
           
            self.knowledges = [" ".join(random.sample(k["knowledges"], min(3, len(k["knowledges"])))) for k in knowledges]
            self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            # load label
            with open(l_path, 'r') as file:
                self.answers = [line.strip() for line in file.readlines()]
                
        elif self.task == "csqa":
            if self.baseline == "gkp":
                if self.is_train:
                    with open("./knowledges/gkp_gpt3_knowledge/knowledge_gkp_gpt3curie.train.csqa.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                else:
                    with open("./knowledges/gkp_gpt3_knowledge/gkp/csqa/knowledge_gkp_gpt3curie.dev.csqa.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
            questions = [item['question']['stem'] for item in self.data]
            self.knowledges = [" ".join(random.sample(k["knowledges"], min(3, len(k["knowledges"])))) for k in knowledges]
            self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                            if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
           
        elif self.task == "qasc":
            if self.baseline == "gkp":
                if self.is_train:
                    with open("./knowledges/gkp_gpt3_knowledge/knowledge_gkp_gpt3curie.train.qasc.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                else:
                    with open("./knowledges/gkp_gpt3_knowledge/gkp/qasc/knowledge_gkp_gpt3curie.dev.qasc.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
                    
            questions = [item['question']['stem'] for item in self.data]
            self.knowledges = [" ".join(random.sample(k["knowledges"], min(3, len(k["knowledges"])))) for k in knowledges]
            self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            self.answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                            if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                            if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                            if line['answerKey'] == "G" else 7 for line in self.data]
        
     
        elif self.task == "arc-h":
            # test file 없다.
            arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3}
            
            if self.baseline == "gkp":
                if self.is_train:
                    with open("./knowledges/gkp_gpt3_knowledge/knowledge_gkp_gpt3curie.train.arc_h.json", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                else:
                    with open("./knowledges/gkp_gpt3_knowledge/", "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
            with jsonlines.open(self.q_path) as f:
                for line in f.iter():
                    self.data.append(line)
            print(f'len: {len(self.data)}')
            
            self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                  
                    questions = self.data["question"]
                    
            
        elif self.task == "obqa":
            # test file 없다.
            if self.is_train:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="train")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data['question_stem']
                    # self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="test")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data['question_stem']
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "wngr":
            if self.is_train:
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, back_ratio_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        

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
            max_length=512
        )
        
        return encoding
    
    def huggingface_formatting(self):
        """
        에러가 존재하여, ARC challgenge는 별도의 함수를 구축하여, 처리한다.
        """
        idx = 0
        prompts = []
        final_answer = []
        final_knowledges = []
        final_dir_answer = []
        for item in self.data:
            choices = item['choices']['text']
            if self.is_ours:
                question = self.question[idx]
            else:
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
                if self.is_ours:
                    final_knowledges.append(self.knowledges[idx])
                    if not self.is_train:
                        final_dir_answer.append(self.answer[idx])
                prompts.append(preprocess(formatted_prompt, self.tokenizer.eos_token))
                idx += 1
            else:
                idx += 1
                continue
                # formatted_prompt = template.format(
                #     question=question,
                #     sol1=choices[0] if len(choices) > 0 else "",
                #     sol2=choices[1] if len(choices) > 1 else "",
                #     sol3=choices[2] if len(choices) > 2 else "",
                #     sol4=choices[3] if len(choices) > 3 else ""
                # )
                
            
        self.answers = final_answer
        self.knowledges = final_knowledges
        if not self.is_train:
            self.answer = final_dir_answer
        return prompts
    
    def prepare_sft_dataset(self, verbose=True):
        if self.task == "piqa":
            prompts = [preprocess(piqa_instruction.format(question=ques, sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
        elif self.task == "csqa":
            prompts = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                            sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]

        elif self.task == "stqa":
            prompts = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]

        elif self.task == "qasc":
            prompts = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]

        elif self.task in ["arc-e", "arc-h", "obqa"]:
            prompts = self.huggingface_formatting()
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            if self.is_ours:
                prompts = [preprocess(siqa_instruction.format(question=ques, sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(siqa_instruction.format(question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "wngr":
            if self.is_ours:
                prompts = [preprocess(wngr_instruction.format(question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item in self.data]
                    
        self.labels = [int(item) for item in self.answers]
        self.encoding = self.tokenizing(prompts)
        
        # torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {prompts[0]}')
            print(self.labels[0])

        return None
    
    def validation(self):
        # inputs
        if self.task == "piqa":
            self.inputs = [preprocess(piqa_instruction.format(question=f'{item["goal"].strip()} {ques}', sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]

        elif self.task == "csqa":
            self.inputs = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
    
        elif self.task == "stqa":
            self.inputs = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]
          
        elif self.task == "qasc":
            self.inputs = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
        
        elif self.task in ["arc-e", "arc-h", "obqa"]:
            self.inputs = self.huggingface_formatting()
            
            print(f'second # of answer: {len(self.answers)}')
     
        elif self.task == "wngr":
            self.inputs = [preprocess(wngr_instruction.format(question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            
        # labels
        self.labels = [int(item) for item in self.answers]
        
        
        print(self.inputs[0])
        print(self.labels[0])
        return None
    
    def __getitem__(self, index):
        return {
            "input_ids": self.encoding["input_ids"][index],
            "attention_mask": self.encoding["attention_mask"][index],
            "labels": torch.tensor(self.labels[index], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)  
        
        
        
        
        
        
        
if __name__ == "__main__":
    """"
    Instruction model에도 INST라는 토큰은 없다.
    """
    tokenizer = AutoTokenizer.from_pretrained("allenai/unifiedqa-t5-large")
    special_tokens_dict = {}
    if tokenizer.pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
        print("add pad token")
    if tokenizer.eos_token is None:
        special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
        print("add eos token")
    if tokenizer.bos_token is None:
        special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
        print("add bos token")
        None
    if tokenizer.unk_token is None:
        special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
        print("add unk token")

    token_list = ['[INST]', "[/INST]"]
    special_tokens_dict['additional_special_tokens'] = token_list
    tokenizer.add_special_tokens(special_tokens_dict)

    print(tokenizer.convert_ids_to_tokens(50257))
    print(tokenizer.convert_ids_to_tokens(50259))
    print(tokenizer.eos_token)
    # tokenizer.padding_side = "right" # standard
    data = CorrectionJSONDataset("wngr", "./dataset/stqa/dev.json",
                                 "./dataset/stqa/dev.json", tokenizer, True, False, "./dataset/stqa/converted_dev.json")
    print(data[0])
    
    
    
    # def postprocessing_prev(self, knowledges, ratio):
    #     self.answer_knowledge = [knowledge.get("Answer_Knowledge", knowledge.get("Background_Knowledge", "Only consider the given question")).strip() for knowledge in knowledges]
    #     self.back_knowledge = [knowledge.get("Background_Knowledge", knowledge.get("Anwer_Knowledge", "Only consider the given question")).strip() for knowledge in knowledges]
    #     self.confidence = [confidence.get("Confidence", 0) for confidence in knowledges]
    #     self.knowledges = []
        
    #     for idx in range(len(self.answer_knowledge)):
    #         if float(self.confidence[idx]) > ratio:
    #             if self.answer_knowledge[idx] == "Not_Knowledge":
    #                 self.knowledges.append(self.back_knowledge[idx])
    #             else: # related to answer
    #                 self.knowledges.append(self.answer_knowledge[idx])
    #         else:
    #             if self.back_knowledge[idx] == "Not_Knowledge":
    #                 self.knowledges.append(self.answer_knowledge[idx])
    #             else: # related to question
    #                 self.knowledges.append(self.back_knowledge[idx])