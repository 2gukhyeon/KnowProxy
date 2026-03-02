import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
from utils import duplicate
# input format
piqa_instruction = '[physicaliqa]:\n<goal>{question}</goal>\n<sol1>{sol1}</sol1>\n<sol2>{sol2}</sol2>'
csqa_instruction = '[commonsenseqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>'
stqa_instruction = '[strategyqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'
qasc_instruction = '[qasc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>\n<option5>{sol5}</option5>\n<option6>{sol6}</option6>\n<option7>{sol7}</option7>\n<option8>{sol8}</option8>'
arc_instruction = '[arc]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
obqa_instruction = '[obqa]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>\n<option4>{sol4}</option4>'
siqa_instruction = '[siqa]:\n<context>{context}</context>\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>\n<option3>{sol3}</option3>'
wngr_instruction = '[wngr]:\n<question>{question}</question>\n<option1>{sol1}</option1>\n<option2>{sol2}</option2>'
boolq_instruction = '[boolq]:\n<passage>{context}</passage>\n<question>{question}</question>\n<option1>True</option1>\n<option2>False</option2>'
truthfulqa_instruction = '[truthfulqa]:\n<question>{question}</question>'


chemprot_instruction = '[chemprot]:\n<question>{question}</question>\n<option1>INHIBITOR</option1>\n<option2>SUBSTRATE</option2>\n<option3>INDIRECT-UPREGULATOR</option3>\n<option4>INDIRECT-DOWNREGULATOR</option4>\n<option5>ACTIVATOR</option5>\n<option6>ANTAGONIST</option6>\n<option7>PRODUCT-OF</option7>\n<option8>AGONIST</option8>\n<option9>DOWNREGULATOR</option9>\n<option10>UPREGULATOR</option10>\n<option11>AGONIST-ACTIVATOR</option11>\n<option12>SUBSTRATE_PRODUCT-OF</option12>\n<option13>AGONIST-INHIBITOR</option13>'
acl_arc_instruction = '[acl_arc]:\n<question>{question}</question>\n<option1>Background</option1>\n<option2>Motivation</option2>\n<option3>CompareOrContrast</option3>\n<option4>Uses</option4>\n<option5>Extends</option5>\n<option6>Future</option6>'
scicite_instruction = '[scicite]:\n<question>{question}</question>\n<option1>background</option1>\n<option2>method</option2>\n<option3>result</option3>'


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

# For cls
class CorrectionJSONDataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True, is_ours=False, cq_path=None, inference_ratio_value=0.9, direction_ratio_value=0.9, upper_ratio=0.9): 
        self.q_path = q_path
        self.l_path = l_path
        self.data = []
        self.tokenizer = tokenizer
        self.is_ours = is_ours
        self.task = task
        self.is_train = is_train
        ratio_ = inference_ratio_value # 0 ~ 1 setting
        ## 메모: wngr 0.5 로 훈련하고, 1로 테스트하기.
        direction_ratio_value_ = direction_ratio_value
        self.upper_ratio = upper_ratio
        # load question
        if self.task == "piqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_) # original
                # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['goal'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)

            # load label
            with open(l_path, 'r') as file:
                self.answers = [line.strip() for line in file.readlines()]        
        elif self.task == "csqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['question']['stem'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_) # original
                # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal

                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                questions = [item['question'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                self.answers = [1 if line["answer"] else 0 for line in self.data] # True -> 1, False -> 0
            else:
                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.answers = [1 if line["answer"] else 0 for line in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
                questions = [item['question']['stem'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                self.answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                                if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                                if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                                if line['answerKey'] == "G" else 7 for line in self.data]
            else:
                with jsonlines.open(self.q_path) as f:
                    for line in f.iter():
                        self.data.append(line)
            
                self.answers = [0 if line['answerKey'] == "A" else 1 if line['answerKey'] == "B" else 2 
                                if line['answerKey'] == "C" else 3 if line['answerKey'] == "D" else 4 
                                if line['answerKey'] == "E" else 5 if line['answerKey'] == "F" else 6
                                if line['answerKey'] == "G" else 7 for line in self.data]
        elif self.task == "arc-e":
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="train")
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_)
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]

            else:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Easy", split="test")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} Knowledge: {know.strip()}' for ques, know in zip(questions, self.knowledges)]       
        elif self.task == "ko-arc":
            arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3}
            
            if self.is_train:
                self.data = load_dataset("davidkim205/ko_arc_challenge", split="train")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("davidkim205/ko_arc_challenge", split="test")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        
        elif self.task == "arc-h":
            arc_label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3,
                             '1': 0, '2': 1, '3': 2, '4': 3}
            
            if self.is_train:
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="train")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("allenai/ai2_arc", name = "ARC-Challenge", split="test")
                answers = self.data["answerKey"]
                self.answers = [arc_label_map[str(answer)] for answer in answers]
                print(f'first # of answer: {len(self.answers)}')
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        
        elif self.task == "obqa":
            if self.is_train:
                self.data = load_dataset("allenai/openbookqa", name = "main", split="train")
                answers = self.data["answerKey"]
                self.answers = [0 if answer == "A" else 1 if answer == "B" else 2 
                                if answer == "C" else 3 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_) # original
                    # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal
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
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal
                    questions = self.data['question_stem']
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "siqa":
            if self.is_train:
                self.data = load_dataset("allenai/social_i_qa", split="train")
                answers = self.data["label"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("allenai/social_i_qa", split="validation")
                answers = self.data["label"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_) # original
                    # self.postprocessing_wans(knowledges, ratio_, direction_ratio_value_) # rebuttal
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "wngr":
            if self.is_train:
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="train")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("allenai/winogrande", name = "winogrande_xl", split="validation")
                answers = self.data["answer"]
                self.answers = [int(answer)-1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["sentence"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "boolq":
            if self.is_train:
                self.data = load_dataset("google/boolq", split="train")
                answers = self.data["answer"]
            
                self.answers = [0 if answer else 1 for answer in answers]
                
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                self.data = load_dataset("google/boolq", split="validation")
                answers = self.data["answer"]
                self.answers = [0 if answer else 1 for answer in answers]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "mcsqa_ru":
            if self.is_train:
                self.data = load_dataset("yusuke1997/mCSQA","ru", split="train")
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("yusuke1997/mCSQA","ru", split="test")
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        elif self.task == "mcsqa_fr":
            if self.is_train:
                self.data = load_dataset("yusuke1997/mCSQA","fr", split="train")
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else: # validation
                self.data = load_dataset("yusuke1997/mCSQA","fr", split="test")
                self.answers = [0 if line["answerKey"] == "A" else 1 if line["answerKey"] == "B" else 2 
                if line["answerKey"] == "C" else 3 if line["answerKey"] == "D" else 4 for line in self.data]
                if self.is_ours:
                    with open(cq_path, "r", encoding="utf-8") as f:
                        knowledges = json.load(f)
                    self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                    questions = self.data["question"]
                    self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
        
        # domain specific
        elif self.task == "acl_arc":
            label2id = {
                "background": 0,
                "motivation": 1,
                "compareorcontrast": 2,
                "uses": 3,
                "extends": 4,
                "future": 5,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
            self.answers = [label2id[line["label"].lower().strip()] for line in self.data]
        elif self.task == "chemprot":
            label2id = {
                "INHIBITOR": 0,
                "SUBSTRATE": 1,
                "INDIRECT-UPREGULATOR": 2,
                "INDIRECT-DOWNREGULATOR": 3,
                "ACTIVATOR": 4,
                "ANTAGONIST": 5,
                "PRODUCT-OF": 6,
                "AGONIST": 7,
                "DOWNREGULATOR": 8,
                "UPREGULATOR": 9,
                "AGONIST-ACTIVATOR": 10,
                "SUBSTRATE_PRODUCT-OF": 11,
                "AGONIST-INHIBITOR": 12,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question

            self.answers = [label2id[line["label"].upper().strip()] for line in self.data]
        elif self.task == "scicite":
            label2id = {
                "background": 0,
                "method": 1,
                "result": 2,
            }
            
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question
                questions = [item["text"] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            else:
                data =pd.read_json(self.q_path, lines=True, encoding="utf-8")
                self.data = data.to_dict(orient="records") # text: question

            self.answers = [label2id[line["label"].lower().strip()] for line in self.data]
        
        if is_train:
            self.prepare_sft_dataset()
        else:
            self.validation()

    def postprocessing(self, knowledges, ratio, direction_ratio=0.5):
        """
        ICLR version.
        """
        def knowledge_processing(knowledge, direction_ratio=0.5):
            """
            [direction_ratio]
            < 0.5: negative
            = 0.5: orthogonal
            > 0.5: positive
            """
            ans_knowledge = ""
            idx = 1
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:
                    if (cur_knowledge["Confidence"]/100) >= direction_ratio:
                        ans_knowledge += cur_knowledge[f"Knowledge_{idx}"].strip().strip(".") + ". "
                    idx += 1
                
                return ans_knowledge
            except:
                return "Solve the given question accurately. " # instruction type
        
        def knowledge_confidence(knowledge):
            confidence = 1
            count = 0
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:       
                    confidence = confidence * cur_knowledge["Confidence"] / 100
                    count += 1
                return count, confidence
            except:
                confidence = 0
                return count, confidence
            
        def knowledge_confidence_weighted_sum(knowledge):
            confidence = 0
            count = 0
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:       
                    confidence = confidence + (cur_knowledge["Confidence"] / 100)
                    count += 1
                return count, confidence
            except:
                confidence = 0
                return count, confidence
            
        self.knowledges = []
        self.answer = []
        idx = 0
        for knowledge in knowledges:
            idx += 1
            cur_answer = knowledge["Final_Answer"]
            cur_final_answer_confidence = knowledge["Final_Confidence"]
            if cur_final_answer_confidence is None:
                cur_final_answer_confidence = 0
                answer_cnt = 0
            else:
                cur_final_answer_confidence = cur_final_answer_confidence / 100
                answer_cnt = 1
                
            knowledge_cnt, knowledge_confidence_score = knowledge_confidence(knowledge)
            overall_confidence = cur_final_answer_confidence * knowledge_confidence_score # overall confidence (to mitigate overconfidence) # original
            
            ### 1. weighted sum (i.e., mean)
            # knowledge_cnt, knowledge_confidence_score = knowledge_confidence_weighted_sum(knowledge)
            # overall_confidence = cur_final_answer_confidence + knowledge_confidence_score
            # try:
            #     overall_confidence = overall_confidence / (answer_cnt + knowledge_cnt)
            # except:
            #     overall_confidence = 0
            
            # 2. geometric mean to mitigate penalty of length
            # try:
            #     final_confidence = overall_confidence ** (1/(knowledge_cnt + answer_cnt)) 
            # except:
            #     final_confidence = 0 
            # overall_confidence = cur_final_answer_confidence # only answer for analysis
            ####################### train #####################
            if self.is_train:
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                self.knowledges.append(answer_knowledge)
            
            ####################### inference #######################
            if (overall_confidence >= ratio) and (not self.is_train): # original 
            # if (final_confidence >= ratio) and (not self.is_train) # geometric mean
            # if (overall_confidence >= ratio) and (overall_confidence <= self.upper_ratio)and (not self.is_train): # confidence 
                if ((cur_answer == None) or ("none" in cur_answer.lower()) or ("neither" in cur_answer.lower()) or duplicate(cur_answer)): 
                    # print("this instance has null answer") # error
                    answer_knowledge = knowledge_processing(knowledge, direction_ratio) 
                    self.knowledges.append(answer_knowledge)
                    self.answer.append("Not answer")
                else:
                    self.knowledges.append("Direct Answer") # zero-shot 성능 이용.
                    self.answer.append(cur_answer)
            # elif (final_confidence < ratio) and (not self.is_train): # geometric mean
            elif (overall_confidence < ratio) and (not self.is_train):
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                if cur_answer is None:
                    self.answer.append("Not answer") # 오답으로 처리.
                else:
                    self.answer.append(cur_answer)
                self.knowledges.append(answer_knowledge)

    def postprocessing_wans(self, knowledges, ratio, direction_ratio=0.5):
        """
        ICLR rebuttal version.
        """
        def answer_mapping(prediction):
            prediction_lower = prediction.lower()
            if prediction_lower == "true":
                return "sol1"
            elif prediction_lower == "false":
                return "sol2"
            
            for item in prediction:
                if item == "A":
                    return "sol1"
                elif item == "B":
                    return "sol2"
                else:
                    return "Not answer"
                
        answer_label_map = {'A': "option1", "True": "option1", "False": "option2", 'B': "option2", 'C': "option3", 'D': "option4", 'E': "option5", 'F': "option6", 'G': "option7", 'H': "option8",
                            }
        # answer_label_map = {'A': "sol1", 'B': "sol2"}
        def knowledge_processing(knowledge, direction_ratio=0.5):
            """
            [direction_ratio]
            < 0.5: negative
            = 0.5: orthogonal
            > 0.5: positive
            """
            ans_knowledge = ""
            idx = 1
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:
                    if (cur_knowledge["Confidence"]/100) >= direction_ratio:
                        ans_knowledge += cur_knowledge[f"Knowledge_{idx}"] + ". "
                    idx += 1
                
                return ans_knowledge
            except:
                return "Solve the given question accurately. " # instruction type
        
        def knowledge_confidence(knowledge):
            confidence = 1
            count = 0
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:       
                    confidence = confidence * cur_knowledge["Confidence"] / 100
                    count += 1
                return count, confidence
            except:
                confidence = 0
                return count, confidence
            
        self.knowledges = []
        self.answer = []
        idx = 0
        for knowledge in knowledges:
            idx += 1
            cur_answer = knowledge["Final_Answer"]
            cur_final_answer_confidence = knowledge["Final_Confidence"]
            if cur_final_answer_confidence is None:
                cur_final_answer_confidence = 0
                answer_cnt = 0
            else:
                cur_final_answer_confidence = cur_final_answer_confidence / 100
                answer_cnt = 1
                
            knowledge_cnt, knowledge_confidence_score = knowledge_confidence(knowledge)
            
            overall_confidence = cur_final_answer_confidence * knowledge_confidence_score # overall confidence (to mitigate overconfidence)
            # try:
            #     final_confidence = overall_confidence ** (1/(knowledge_cnt + answer_cnt)) # geometric mean to mitigate penalty of length
            # except:
            #     final_confidence = 0 
            # overall_confidence = cur_final_answer_confidence # only answer for analysis
            ####################### train #####################
            if self.is_train:
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                if cur_answer is None:
                    answer_knowledge += f'<model_answer>Not answer</model_answer>' 
                else:
                    answer_knowledge += f'<model_answer>{answer_mapping(knowledge["Final_Answer"])}</model_answer>'
                self.knowledges.append(answer_knowledge)
            
            ####################### inference #######################
            if (overall_confidence >= ratio) and (not self.is_train): # original 
            # if (final_confidence >= ratio) and (not self.is_train) # geometric mean
            # if (overall_confidence >= ratio) and (overall_confidence <= self.upper_ratio)and (not self.is_train): # confidence 
                if ((cur_answer == None) or ("none" in cur_answer.lower()) or ("neither" in cur_answer.lower()) or duplicate(cur_answer)): 
                    # print("this instance has null answer") # error
                    answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                    answer_knowledge += f'<model_answer>Not answer</model_answer>' 
                    self.knowledges.append(answer_knowledge)
                    self.answer.append("Not answer")
                else:
                    self.knowledges.append("Direct Answer") # zero-shot 성능 이용.
                    self.answer.append(cur_answer)
            # elif (final_confidence < ratio) and (not self.is_train):
            elif (overall_confidence < ratio) and (not self.is_train):
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                if cur_answer is None:
                    self.answer.append("Not answer") # 오답으로 처리.
                    answer_knowledge += f'<model_answer>Not answer</model_answer>' 
                else:
                    self.answer.append(cur_answer)
                    answer_knowledge += f'<model_answer>{answer_mapping(knowledge["Final_Answer"])}</model_answer>'
                self.knowledges.append(answer_knowledge)
    
    
      
    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            add_special_tokens=True,
            padding=True,
            truncation="longest_first",
            max_length=512 # ours
            # max_length=256 
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
                if self.task in ["arc-e", "arc-h", "ko-arc"]:
                    question = item['question']
                elif self.task == "obqa":
                    question = item['question_stem']
            
            if self.task in ["arc-e", "arc-h", "ko-arc"]:
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
            
        self.answers = final_answer
        self.knowledges = final_knowledges
        if not self.is_train:
            self.answer = final_dir_answer
        return prompts
    
    def prepare_sft_dataset(self, verbose=True):
        if self.task == "piqa":
            if self.is_ours:
                prompts = [preprocess(piqa_instruction.format(question=ques, sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "csqa":
            if self.is_ours:
                prompts = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                prompts = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(stqa_instruction.format(question=item['question'], sol1="True", sol2="False"), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                prompts = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(qasc_instruction.format(question=item['question']['stem'], sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task in ["arc-e", "arc-h", "obqa", "ko-arc"]:
            prompts = self.huggingface_formatting()
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            if self.is_ours:
                prompts = [preprocess(siqa_instruction.format(context=item["context"], question=ques, sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(siqa_instruction.format(context=item["context"],question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "wngr":
            if self.is_ours:
                prompts = [preprocess(wngr_instruction.format(question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "boolq":
            if self.is_ours:
                prompts = [preprocess(boolq_instruction.format(context=item["passage"], question=ques), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(boolq_instruction.format(context=item["passage"], question=item['question']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "chemprot":
            if self.is_ours:
                prompts = [preprocess(chemprot_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(chemprot_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "acl_arc":
            if self.is_ours:
                prompts = [preprocess(acl_arc_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(acl_arc_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]        
        elif self.task == "scicite":
            if self.is_ours:
                prompts = [preprocess(scicite_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(scicite_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]              
        elif self.task in ["mcsqa_ru", "mcsqa_fr"]:
            if self.is_ours:
                prompts = [preprocess(csqa_instruction.format(question=ques, sol1=item["choices"]['text'][0], sol2=item["choices"]['text'][1], sol3=item["choices"]['text'][2], 
                                                              sol4=item["choices"]['text'][3], sol5=item["choices"]['text'][4]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(csqa_instruction.format(question=item['question'], sol1=item["choices"]['text'][0], sol2=item["choices"]['text'][1], sol3=item["choices"]['text'][2], 
                                                              sol4=item["choices"]['text'][3], sol5=item["choices"]['text'][4]), self.tokenizer.eos_token) for item in self.data]     
        
        self.labels = [int(item) for item in self.answers]
        self.encoding = self.tokenizing(prompts)
        
        # torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {prompts[0]}')
            print(self.labels[0])

        return None
    
    def validation(self, verbose=False):
        # inputs
        if self.task == "piqa":
            if self.is_ours:
                self.inputs = [preprocess(piqa_instruction.format(question=f'{item["goal"].strip()} {ques}', sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(piqa_instruction.format(question=item["goal"], sol1=item["sol1"], sol2=item["sol2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "csqa":
            if self.is_ours:
                self.inputs = [preprocess(csqa_instruction.format(question=ques, sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(csqa_instruction.format(question=item['question']['stem'], sol1=item["question"]["choices"][0]['text'], sol2=item["question"]["choices"][1]['text'], sol3=item["question"]["choices"][2]['text'], 
                                                              sol4=item["question"]["choices"][3]['text'], sol5=item["question"]["choices"][4]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "stqa":
            if self.is_ours:
                self.inputs = [preprocess(stqa_instruction.format(question=ques, sol1="True", sol2="False"), self.tokenizer.eos_token) for _, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(stqa_instruction.format(question=item['question'], sol1="True", sol2="False"), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "qasc":
            if self.is_ours:
                self.inputs = [preprocess(qasc_instruction.format(question=ques, sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(qasc_instruction.format(question=item['question']['stem'], sol1=item['question']['choices'][0]['text'], sol2=item['question']['choices'][1]['text'], sol3=item['question']['choices'][2]['text'], sol4=item['question']['choices'][3]['text'],
                                                              sol5=item['question']['choices'][4]['text'], sol6=item['question']['choices'][5]['text'], sol7=item['question']['choices'][6]['text'], sol8=item['question']['choices'][7]['text']), self.tokenizer.eos_token) for item in self.data]
        elif self.task in ["arc-e", "arc-h", "obqa", "ko-arc"]:
            self.inputs = self.huggingface_formatting()  
            print(f'second # of answer: {len(self.answers)}')
        elif self.task == "siqa":
            if self.is_ours:
                self.inputs = [preprocess(siqa_instruction.format(context=item["context"], question=ques, sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(siqa_instruction.format(context=item["context"], question=item['question'], sol1=item["answerA"], sol2=item["answerB"], sol3=item["answerC"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "wngr":
            if self.is_ours:
                self.inputs = [preprocess(wngr_instruction.format(question=ques, sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(wngr_instruction.format(question=item['sentence'], sol1=item["option1"], sol2=item["option2"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "boolq":
            if self.is_ours:
                self.inputs = [preprocess(boolq_instruction.format(context=item["passage"], question=ques), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(boolq_instruction.format(context=item["passage"], question=item['question']), self.tokenizer.eos_token) for item in self.data] 
        elif self.task == "chemprot":
            if self.is_ours:
                self.inputs = [preprocess(chemprot_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(chemprot_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]          
        elif self.task == "acl_arc":
            if self.is_ours:
                self.inputs = [preprocess(acl_arc_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(acl_arc_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]   
        elif self.task == "scicite":
            if self.is_ours:
                self.inputs = [preprocess(scicite_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                self.inputs = [preprocess(scicite_instruction.format(question=item['text']), self.tokenizer.eos_token) for item in self.data]              
        elif self.task in ["mcsqa_ru", "mcsqa_fr"]:
            if self.is_ours:
                self.inputs = [preprocess(csqa_instruction.format(question=ques, sol1=item["choices"]['text'][0], sol2=item["choices"]['text'][1], sol3=item["choices"]['text'][2], 
                                                              sol4=item["choices"]['text'][3], sol5=item["choices"]['text'][4]), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(csqa_instruction.format(question=item['question'], sol1=item["choices"]['text'][0], sol2=item["choices"]['text'][1], sol3=item["choices"]['text'][2], 
                                                              sol4=item["choices"]['text'][3], sol5=item["choices"]['text'][4]), self.tokenizer.eos_token) for item in self.data]     
        
        
        # labels
        self.labels = [int(item) for item in self.answers]
        
        if verbose:
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
        
    # def __getitem__(self, index):
    #     data = {key: val[index] for key, val in self.encoding.items()}
    #     data['labels'] = self.labels[index]
    #     return data 

    # def __len__(self) -> int:
    #     return len(self.labels)

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
    data = CorrectionJSONDataset("boolq", "./dataset/boolq/converted_train.json",
                                 "./dataset/boolq/converted_train.json", tokenizer, True,True, "./dataset/boolq/converted_train.json", ratio_value=0, direction_ratio_value_value=0)
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