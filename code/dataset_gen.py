import jsonlines
from torch.utils.data import Dataset
from preprocess import preprocess
from transformers import AutoTokenizer
import torch 
import pandas as pd
import json
from datasets import load_dataset
from utils import duplicate
import re

IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'
GSM8K_COT_question = ["Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
             "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
             "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
             "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
             "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
             "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
             "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
             "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
             ]

# GSM8K_COT_question_knowproxy = ["Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?<knowledge></knowledge>",
#              "Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
#              "Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
#              "Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
#              "Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
#              "Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
#              "Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
#              "Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?"
#              ]
GSM8K_COT_answer = ["Let's think step by step.\nThere are 15 trees originally.\nThen there were 21 trees after some more were planted.\nSo there must have been 21 - 15 = 6.\nThe answer is 6\n\n<|eot_id|>",
                    "Let's think step by step.\nThere are originally 3 cars.\n2 more cars arrive.\n3 + 2 = 5.\nThe answer is 5\n\n<|eot_id|>",
                    "Let's think step by step.\nOriginally, Leah had 32 chocolates.\nHer sister had 42.\nSo in total they had 32 + 42 = 74.\nAfter eating 35, they had 74 - 35 = 39.\nThe answer is 39\n\n<|eot_id|>",
                    "Let's think step by step.\nJason started with 20 lollipops.\nThen he had 12 after giving some to Denny.\nSo he gave Denny 20 - 12 = 8.\nThe answer is 8\n\n<|eot_id|>",
                    "Let's think step by step.\nShawn started with 5 toys.\nIf he got 2 toys each from his mom and dad, then that is 4 more toys.\n5 + 4 = 9.\nThe answer is 9\n\n<|eot_id|>",
                    "Let's think step by step.\nThere were originally 9 computers.\nFor each of 4 days, 5 more computers were added.\nSo 5 * 4 = 20 computers were added.\n9 + 20 is 29.\nThe answer is 29\n\n<|eot_id|>",
                    "Let's think step by step.\nMichael started with 58 golf balls.\nAfter losing 23 on tues- day, he had 58 - 23 = 35.\nAfter losing 2 more, he had 35 - 2 = 33 golf balls.\nThe answer is 33\n\n<|eot_id|>",
                    "Let's think step by step.\nOlivia had 23 dollars.\n5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.\nSo she has 23 - 15 dollars left.\n23 - 15 is 8.\nThe answer is 8\n\n<|eot_id|>"
                    ]
# GSM8K_COT_question = ["Question: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?",
#              "Question: James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?",
#              "Question: Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?",
#              "Question: Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?",
#              "Question: Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?",
#              "Question: The profit from a business transaction is shared among 2 business partners, Mike and Johnson in the ratio 2:5 respectively. If Johnson got $2500, how much will Mike have after spending some of his share on a shirt that costs $200?",
#              "Question: Tim rides his bike back and forth to work for each of his 5 workdays. His work is 20 miles away. He also goes for a weekend bike ride of 200 miles. If he can bike at 25 mph how much time does he spend biking a week?",
#              "Question: Bella bought stamps at the post office. Some of the stamps had a snowflake design, some had a truck design, and some had a rose design. Bella bought 11 snowflake stamps. She bought 9 more truck stamps than snowflake stamps, and 13 fewer rose stamps than truck stamps. How many stamps did Bella buy in all?"
#              ]
# GSM8K_COT_answer = ["Let's think step by step.\nMaila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\nThe answer is 42\n\n<|eot_id|>",
#                     "Let's think step by step.\nHe writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\nThe answer is 624\n\n<|eot_id|>",
#                     "Let's think step by step.\nHe eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32\nHe eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16\nHe eats 48 pieces\nbecause 32 + 16 = <<32+16=48>>48\nThe answer is 48\n\n<|eot_id|>",
#                     "Let's think step by step.\nHalf of the number of Randy's mango trees is 60/2 = <<60/2=30>>30 trees.\nSo Randy has 30 - 5 = <<30-5=25>>25 coconut trees.\nTherefore, Randy has 60 + 25 = <<60+25=85>>85 treeson his farm.\nThe answer is 85\n\n<|eot_id|>",
#                     "Let's think step by step.\nA pound of cream cheese cost $10 / 2 = $<<10/2=5>>5.\nA pack of cold cuts cost $10 x 2 = $<<10*2=20>>20.\nJasper spent $10 + $5 + $20 = $<<10+5+20=35>>35 on the ingredients.\nThe answer is 35\n\n<|eot_id|>",
#                     "Let's think step by step.\nAccording to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts\nSince Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500\nMike will get 2*$500 = $<<2*500=1000>>1000\nAfter buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left\nThe answer is 800\n\n<|eot_id|>",
#                     "Let's think step by step.\nHe bikes 20*2=<<20*2=40>>40 miles each day for work\nSo he bikes 40*5=<<40*5=200>>200 miles for work\nThat means he bikes a total of 200+200=<<200+200=400>>400 miles for work\nSo he bikes a total of 400/25=<<400/25=16>>16 hours\nThe answer is 16\n\n<|eot_id|>",
#                     "Let's think step by step.\nThe number of truck stamps is 11 + 9 = <<11+9=20>>20.\nThe number of rose stamps is 20 − 13 = <<20-13=7>>7.\nBella bought 11 + 20 + 7 = <<11+20+7=38>>38 stamps in all.\nThe answer is 38\n\n<|eot_id|>"
#                     ]

# GSM8K_COT_answer = ["Let's think step by step.\nMaila read 12 x 2 = <<12*2=24>>24 pages today.\nSo she was able to read a total of 12 + 24 = <<12+24=36>>36 pages since yesterday.\nThere are 120 - 36 = <<120-36=84>>84 pages left to be read.\nSince she wants to read half of the remaining pages tomorrow, then she should read 84/2 = <<84/2=42>>42 pages.\nThe answer is 42",
#                     "Let's think step by step.\nHe writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means he writes 12*52=<<12*52=624>>624 pages a year\nThe answer is 624",
#                     "Let's think step by step.\nHe eats 32 from the largest pizzas because 2 x 16 = <<2*16=32>>32\nHe eats 16 from the small pizza because 2 x 8 = <<2*8=16>>16\nHe eats 48 pieces\nbecause 32 + 16 = <<32+16=48>>48\nThe answer is 48",
#                     "Let's think step by step.\nHalf of the number of Randy's mango trees is 60/2 = <<60/2=30>>30 trees.\nSo Randy has 30 - 5 = <<30-5=25>>25 coconut trees.\nTherefore, Randy has 60 + 25 = <<60+25=85>>85 treeson his farm.\nThe answer is 85",
#                     "Let's think step by step.\nA pound of cream cheese cost $10 / 2 = $<<10/2=5>>5.\nA pack of cold cuts cost $10 x 2 = $<<10*2=20>>20.\nJasper spent $10 + $5 + $20 = $<<10+5+20=35>>35 on the ingredients.\nThe answer is 35",
#                     "Let's think step by step.\nAccording to the ratio, for every 5 parts that Johnson gets, Mike gets 2 parts\nSince Johnson got $2500, each part is therefore $2500/5 = $<<2500/5=500>>500\nMike will get 2*$500 = $<<2*500=1000>>1000\nAfter buying the shirt he will have $1000-$200 = $<<1000-200=800>>800 left\nThe answer is 800",
#                     "Let's think step by step.\nHe bikes 20*2=<<20*2=40>>40 miles each day for work\nSo he bikes 40*5=<<40*5=200>>200 miles for work\nThat means he bikes a total of 200+200=<<200+200=400>>400 miles for work\nSo he bikes a total of 400/25=<<400/25=16>>16 hours\nThe answer is 16",
#                     "Let's think step by step.\nThe number of truck stamps is 11 + 9 = <<11+9=20>>20.\nThe number of rose stamps is 20 − 13 = <<20-13=7>>7.\nBella bought 11 + 20 + 7 = <<11+20+7=38>>38 stamps in all.\nThe answer is 38"
#                     ]

# instruction = "### Question\n{question}\nLet's think step by step."
instruction = "Question: {question}"
aqua_instruction = "{question}\n{opt1}\n{opt2}\n{opt3}\n{opt4}\n{opt5}"
truthfulqa_instruction = '{question}'
scienceqa_instruction = 'Question: {question}\nChoices:'
# PROMPT = '''
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. 
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
# '''.strip() + "\n"
# truthfulqa_instruction = '{PROMPT}\nQ: {question}\nA:' # BBox-Adapter ver.
class Gen_Dataset(Dataset):
    def __init__(self, task, q_path, l_path, tokenizer, is_train=True, is_ours=False, cq_path=None, inference_ratio_value=0.9, direction_ratio_value=0.9,): 
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
        
        # load question
        if task == "gsm8k":
            if self.is_train:
                self.data = load_dataset("openai/gsm8k", name = "main", split="train")
                
            else: # validation
                self.data = load_dataset("openai/gsm8k", name = "main", split="test")
            
            self.answers = [line["answer"].replace('####', 'The answer is').strip() for line in self.data]
            self.answers = [re.sub(f'<<.*?>>', '', line) for line in self.answers]
                
           
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                questions = self.data["question"]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
                
        elif self.task == "truthfulqa":
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                
                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                questions = [item['Question'] for item in self.data]
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            
                self.answers = [line["Best Answer"] for line in self.data]
            else:
                with open(self.q_path, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
                self.answers = [line["Best Answer"] for line in self.data]
        elif self.task == "scienceqa":
            answer_map = {"0": "A", "1": "B", "2": "C", "3": "D",
                          "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J"}
            with open(self.q_path, "r", encoding="utf-8") as f:
                self.data = json.load(f) # subsampling dataset
                
            self.answers = [answer_map[str(line["answer"])] for line in self.data]
 
            if self.is_ours:
                with open(cq_path, "r", encoding="utf-8") as f:
                    knowledges = json.load(f)
                self.postprocessing(knowledges, ratio_, direction_ratio_value_)
                questions = [line["question"] for line in self.data]
             
                self.question = [f'{ques.strip()} <knowledge>{know.strip()}</knowledge>' for ques, know in zip(questions, self.knowledges)]
            
        elif task == "aqua":
            if self.is_train:
                self.data = load_dataset("deepmind/aqua_rat", name = "raw", split="train")
                self.answers = self.data["rationale"]
                print(f'# of data: {len(self.data)}')
            else: # validation
                self.data = load_dataset("deepmind/aqua_rat", name = "raw", split="test")
                self.answers = self.data["correct"]
        
            
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
            try:
                for cur_knowledge in knowledge["Answer_Knowledge"]:       
                    confidence = confidence * cur_knowledge["Confidence"] / 100
                return confidence
            except:
                confidence = 0
                return confidence
            
        self.knowledges = []
        self.answer = []
        idx = 0
        for knowledge in knowledges:
            idx += 1
            cur_answer = knowledge["Final_Answer"]
            cur_final_answer_confidence = knowledge["Final_Confidence"]
            if cur_final_answer_confidence is None:
                cur_final_answer_confidence = 0
            else:
                cur_final_answer_confidence = cur_final_answer_confidence / 100
                
            knowledge_confidence_score = knowledge_confidence(knowledge)
            overall_confidence = cur_final_answer_confidence * knowledge_confidence_score # overall confidence는 모든 출력물의 평균이다. (to mitigate overconfidence)
            # overall_confidence = cur_final_answer_confidence # only answer
            ####################### train #####################
            if self.is_train:
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                self.knowledges.append(answer_knowledge)
            
            ####################### inference #######################
            if (overall_confidence >= ratio) and (not self.is_train):
            # if (overall_confidence >= ratio) and (overall_confidence <= self.upper_ratio)and (not self.is_train): # confidence 
                if ((cur_answer == None) or ("none" in cur_answer.lower()) or ("neither" in cur_answer.lower()) or duplicate(cur_answer)): 
                    # print("this instance has null answer") # error
                    answer_knowledge = knowledge_processing(knowledge, direction_ratio) 
                    self.knowledges.append(answer_knowledge)
                    self.answer.append("Not answer")
                else:
                    self.knowledges.append("Direct Answer") # zero-shot 성능 이용.
                    self.answer.append(cur_answer)
            elif (overall_confidence < ratio) and (not self.is_train):
                answer_knowledge = knowledge_processing(knowledge, direction_ratio)
                if cur_answer is None:
                    self.answer.append("Not answer") # 오답으로 처리.
                else:
                    self.answer.append(cur_answer)
                self.knowledges.append(answer_knowledge)
    
    def tokenizing(self, lines):
        encoding = self.tokenizer(
            lines,
            return_tensors = "pt",
            add_special_tokens=False, # False; then use chat template
            padding="longest",
            truncation=True,
            max_length=2048 # generative tasks
        )
        
        return encoding
    
    def prepare_sft_dataset(self, verbose=True):
        prompts = []
        if self.task == "gsm8k":
            if self.is_ours:
                for ques in self.question:
                    content = [
                        {
                            "role": "user",
                            # "content": GSM8K_COT + "\n\n" + instruction.format(question=ques["question"].strip())
                            "content": instruction.format(question=ques.strip())                        
                        }
                    ]    
                    prompts.append(self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)) 
            else:
                for ques in self.data:
                    content = [
                        {
                            "role": "user",
                            # "content": GSM8K_COT + "\n\n" + instruction.format(question=ques["question"].strip())
                            "content": instruction.format(question=ques["question"].strip())                        
                        }
                    ]
          
                    prompts.append(self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True))
                
        

        elif self.task == "aqua":
            if self.is_ours:
                prompts = [preprocess(aqua_instruction.format(question=ques, opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for ques, item in zip(self.question, self.data)]
            else:
                prompts = [preprocess(aqua_instruction.format(question=item["question"], opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "truthfulqa":
            if self.is_ours:
                prompts = [preprocess(truthfulqa_instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            else:
                prompts = [preprocess(truthfulqa_instruction.format(question=item["Question"]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "scienceqa":
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            def format_option(targets):
                options = []
                for i, key in enumerate(targets):
                    # options.append(f"<option{i}>{labels[i]}. {key}</option{i}>")
                    options.append(f"{labels[i]}. {key}")
                return '\n'.join(options)   
            if self.is_ours:
                prompts = [preprocess(scienceqa_instruction.format(question=ques) + '\n' + format_option(item['choices']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                prompts = [preprocess(scienceqa_instruction.format(question=item["question"]) + '\n' + format_option(item['choices']), self.tokenizer.eos_token) for item in self.data]

        
        answers = [f"Let's think step by step.\n{item}" for item in self.answers]
        # texts = [str(prompt) + str(answer) + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)] # original
        # texts = [str(prompt) + answer + self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)] # ref. Ualign
        # answers = [item for item in self.answers]
        texts = [prompt + answer + "\n\n"+ self.tokenizer.eos_token for prompt, answer in zip(prompts, answers)]

        self.encoding = self.tokenizing(texts)
        input_ids = self.encoding["input_ids"]
        self.labels = input_ids.clone()
        

        for idx in range(len(self.labels)):
            self.labels[idx,:len(self.tokenizing(prompts[idx])['input_ids'][0])] = IGNORE_INDEX
        
        torch.set_printoptions(profile="full") # for code verifying
        if verbose:
            print(f'sample example: {texts[0]}')
            filtered = self.labels[0][self.labels[0] != -100]
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
            print(tokenizer.decode(filtered, skip_special_tokens=False))
            # print(f'prompt:\n{prompts[0]}')
            # print(f'prompt:\n{str(answers[0])}')
            # print(input_ids[0])
            # print(self.tokenizing(prompts[0]["input_ids"]))
        return None
    
    def validation(self):
        self.inputs = []
        if self.task == "gsm8k":
            if self.is_ours:
                for ques in self.question:
                    content = []
                    # examples
                    for question, answer in zip(GSM8K_COT_question, GSM8K_COT_answer):
                        content.append({"role": "user", "content": question})
                        content.append({"role": "assistant", "content": answer})
                    content.append({
                            "role": "user",
                            "content": instruction.format(question=ques.strip()) 
                        })
                  
                    prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
        
                    prompt = prompt.replace("<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>", "<|eot_id|>\n<|eot_id|><|start_header_id|>user<|end_header_id|>")
                    self.inputs.append(prompt)
            else:
                for ques in self.data:
                    content = []
                    # examples
                    for question, answer in zip(GSM8K_COT_question, GSM8K_COT_answer):
                        content.append({"role": "user", "content": question})
                        content.append({"role": "assistant", "content": answer})
                    content.append({
                            "role": "user",
                            "content": instruction.format(question=ques["question"].strip()) 
                        })
                    # ]
                    prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
        
                    prompt = prompt.replace("<|eot_id|><|eot_id|><|start_header_id|>user<|end_header_id|>", "<|eot_id|>\n<|eot_id|><|start_header_id|>user<|end_header_id|>")
                    self.inputs.append(prompt)
                    
            # if self.is_ours:
            #     self.inputs = [preprocess(instruction.format(question=ques), self.tokenizer.eos_token) for ques in self.question]
            # else:
            #     self.inputs = [preprocess(instruction.format(question=item["question"].strip()), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "aqua":
            if self.is_ours:
                self.inputs = [preprocess(aqua_instruction.format(question=ques, opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for ques, item in zip(self.question, self.data)]
            else:
                self.inputs = [preprocess(aqua_instruction.format(question=item["question"], opt1=item["options"][0], opt2=item["options"][1], opt3=item["options"][2], opt4=item["options"][3], opt5=item["options"][4]), self.tokenizer.eos_token) for item in self.data]
        elif self.task == "scienceqa":
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            def format_option(targets):
                options = []
                for i, key in enumerate(targets):
                    options.append(f"<option{i}>{labels[i]}. {key}</option{i}>")
                return '\n'.join(options)   
            if self.is_ours:
                self.inputs = [preprocess(scienceqa_instruction.format(question=ques) + '\n' + format_option(item['choices']), self.tokenizer.eos_token) for item, ques in zip(self.data, self.question)]
            else:
                self.inputs = [preprocess(scienceqa_instruction.format(question=item["question"]) + '\n' + format_option(item['choices']), self.tokenizer.eos_token) for item in self.data]
        
        # labels
        self.labels = self.answers
        
        print(self.inputs[0])
        
        return None
        
        
    def __getitem__(self, index):
        data = {key: val[index] for key, val in self.encoding.items()}
        # data['labels'] = self.labels['input_ids'][index]
        data['labels'] = self.labels[index]
        return data 

    def __len__(self) -> int:
        return len(self.labels)



if __name__ == "__main__":
    """"
    Instruction model에도 INST라는 토큰은 없다.
    """
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
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

    # print(tokenizer.convert_tokens_to_ids("[INST]"))
    # print(tokenizer.convert_tokens_to_ids("[/INST]"))
    # print(tokenizer.convert_ids_to_tokens(3297))
    

    # tokenizer.padding_side = "right" # standard
    data = Gen_Dataset("aqua", None, None, tokenizer, True, True, "./dataset/aqua/converted_train.json", 0, 0)
    print(data[0])