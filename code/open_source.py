from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, BitsAndBytesConfig
# from dataset import CorrectionJSONDataset
from dataset_gen import Gen_Dataset, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from dataset_proxy import ProxyDataset
from peft import PeftConfig, PeftModelForCausalLM, PeftModelForSeq2SeqLM
from transformers.utils import logging
logging.set_verbosity_error()
use_devices = "cuda" if torch.cuda.is_available() else "cpu"
pd.set_option("display.max_colwidth", None)

# utils
def build_chat_batch(tokenizer, prompts: list[str], device, max_input_len=3072):
    # 시스템/유저 메시지로 이루어진 chat 템플릿 생성
    conversations = [
        [{"role": "system", "content": "You are a helpful math tutor. Think step by step and show your reasoning clearly."},
         {"role": "user", "content": p}]
        for p in prompts
    ]
    encoded = tokenizer.apply_chat_template(
        conversations,
        add_generation_prompt=True,   # assistant: 의 시작까지 붙여줌
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_len
    )
    return {k: v.to(device) for k, v in {"input_ids": encoded, "attention_mask": (encoded != tokenizer.pad_token_id)}.items()}


class Open_source_LM:
    def __init__(self, model_name="kogpt", is_generate=False, is_lora=True, is_quantized=False, is_seq2seq=False, num_labels=2, device=use_devices):
        """
        ### Argument ### 
        model_name: loading language model.
        is_generate: setting generative model.
        is_lora: setting LoRA.
        is_seq2seq: setting encoder-decoder model.
        num_labels: setting classifier.
        device: GPU or CPU.
        """
        self.model_name = model_name
        self.device = device
        # model
        self.model = None
        self.tokenizer = None
        self.is_lora = is_lora
        self.is_quantized = is_quantized
        self.is_generate = is_generate
        self.is_seq2seq = is_seq2seq
        self.num_labels = num_labels
        
        # templates
        self.free_form_template, self.yes_no_template = None, None

        self.get_default_batch_sizes()

        self.use_tqdm = True
    
    def tqdm(self, *args, **kwargs):
        if self.use_tqdm:
            return tqdm(*args, **kwargs)
        else:
            return args[0]

    def get_default_batch_sizes(self):
        if not torch.cuda.is_available():
            self.inference_batch_size = 8
            return
        # get total memory
        # initialize total_memory
        total_memory = 0

        # iterate over all devices
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory

        # if over 80GB (a100)
        if total_memory > 80_000_000_000:
            self.inference_batch_size = 2056
        # else, if over 50GB (a6000)
        elif total_memory > 50_000_000_000:
            self.inference_batch_size = 64
            # self.inference_batch_size = 1
        else:
            # default
            self.inference_batch_size = 128
            
    def load_model(self):
        print('Loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.is_generate:
            self.tokenizer.padding_side = "left" # original
            self.tokenizer.truncation_side = "left" # original
            # self.tokenizer.padding_side = "right" 
            # self.tokenizer.truncation_side = "right"
        else: # classifier
            self.tokenizer.padding_side = "right"
        
        if self.is_lora:
            config = PeftConfig.from_pretrained(self.model_name)
            if self.is_generate:
                from peft import PeftModel
                # quantized_model_config = BitsAndBytesConfig(  
                #             load_in_4bit= True,
                #             bnb_4bit_quant_type= "nf4",
                #             bnb_4bit_compute_dtype= torch.bfloat16,
                #             bnb_4bit_use_double_quant= True,
                #         )
                
                self.model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                                #   quantization_config=quantized_model_config,
                                                                #   low_cpu_mem_usage=True,
                                                                  torch_dtype=torch.bfloat16
                                                                  )
                ## original ##
                
                
                self.model.resize_token_embeddings(max(len(self.tokenizer), self.model.model.embed_tokens.num_embeddings)) # change embedding size
                self.model = PeftModelForCausalLM.from_pretrained(self.model, model_id=self.model_name)
                self.model = self.model.merge_and_unload()
                self.model.config.pad_token_id = self.tokenizer.pad_token_id
                ## ##
                
           
                # self.model = PeftModel.from_pretrained(
                #     self.model,
                #     model_id=self.model_name)
            else:
                if self.is_seq2seq:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
                    
                    self.model.resize_token_embeddings(len(self.tokenizer)) # change embedding size (-> 32110, 4096)
                    self.model = PeftModelForSeq2SeqLM.from_pretrained(self.model, model_id=self.model_name)
                else:
                    if self.is_quantized:
                        quantized_model_config = BitsAndBytesConfig(  
                            load_in_4bit= True,
                            bnb_4bit_quant_type= "nf4",
                            bnb_4bit_compute_dtype= torch.bfloat16,
                            bnb_4bit_use_double_quant= True,
                        )
                        self.model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=self.num_labels,
                                                                                        quantization_config=quantized_model_config,
                                                                                        low_cpu_mem_usage=True,
                                                                                        # torch_dtype=torch.float16
                                                                                        )
                    else:
                        self.model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=self.num_labels)
                    self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
                    self.model.resize_token_embeddings(len(self.tokenizer)) # change embedding size
                    self.model = PeftModelForCausalLM.from_pretrained(self.model, model_id=self.model_name)
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            # self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            

        if torch.cuda.device_count() > 1:
            self.model.parallelize()
            # self device is model device
            self.device = self.model.device
        else:
            self.model = self.model.to(self.device)
        print("loaded FT model")
        # to device
        self.device = self.model.device

        # eval. mode
        self.model.eval()
       
    def get_output(self, action):
        gens = self.get_labels(action)

        # decode
        parsed = self.parse_labels(gens)
        
        df = self.normalize_preds([act for act in action], [d['pred._label'] for d in parsed])
        return df

    def get_labels(self, actions, batch_size=None):
        if not self.model:
            self.load_model()
        
        is_single = False
        if isinstance(actions, str):
            actions = [actions]
            is_single = True
        
        if batch_size is None:
            batch_size = self.inference_batch_size
        
        batch_outputs = []
        n_batchs = math.ceil(len(actions) / batch_size)

        # for i in range(n_batchs):
        for i in self.tqdm(range(n_batchs), desc="Generations"):
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            
            if self.is_generate:
                encoded_batch = self.tokenizer.batch_encode_plus(
                    [action for action in batch_actions],
                    return_tensors='pt',
                    add_special_tokens=False, # default True
                    padding=True,
                    truncation=True,
                    max_length=2048 # boolq, gsm8k
                    ).to(self.device)
                # encoded_batch = self.tokenizer.batch_encode_plus(
                #     [action for action in batch_actions],
                #     return_tensors='pt',
                #     add_special_tokens=False, # default True
                #     padding=False,
                #     # truncation=True,
                #     # max_length=2048 # boolq, gsm8k
                #     ).to(self.device)
                # print(encoded_batch)
            else:
                encoded_batch = self.tokenizer.batch_encode_plus(
                    [action for action in batch_actions],
                    return_tensors='pt',
                    add_special_tokens=True,
                    padding=True,
                    truncation="longest_first",
                    max_length=512,
                    ).to(self.device)
            
            input_ids = encoded_batch["input_ids"]
            attention_mask = encoded_batch["attention_mask"]
            
            if self.is_generate:
                # generate
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                    do_sample=False, max_new_tokens=256)
               
                # print(outputs)
                for idx in range(len(batch_actions)):
                    outputs[idx,:len(input_ids[idx])] = self.tokenizer.eos_token_id
                    # outputs[idx,:len(self.tokenizer(batch_actions[idx],padding=True,add_special_tokens=True,truncation="longest_first", max_length=1024)['input_ids'])] = self.tokenizer.eos_token_id
                
                # decode
                outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # print(outputs)
                
            else:
                if self.is_seq2seq:
                    with torch.no_grad():
                        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=32)
                    
                    outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                else:
                    # del encoded_batch['token_type_ids']
                    logits = self.model(**encoded_batch).logits
                    
                    #####
                    outputs = logits.argmax(dim=1).tolist() # ours code
                    ###
            
                    # baseline code; 그 외 활성화 끄기.
                    # mean_logits = logits.mean(dim=0)
                    # # 평균 logits이 가장 높은 선지 선택
                    # outputs = mean_logits.argmax().item()
                    # outputs = [outputs for _ in range(len(actions))]
                    
            # add to list
            batch_outputs.extend(outputs)
            # print(batch_outputs)
        # reshape to (n_actions, n_gens)
        decoded = np.array(batch_outputs).reshape((len(actions), 1))

        # strip
        if self.is_generate:
            decoded = np.array([np.array([x.strip() for x in gen]) for gen in decoded])
        
        
        # if single, return as string
        if is_single:
            decoded = decoded[0]

        
        return decoded
    

    
    def parse_label(self, pred):
        if self.is_generate:
            return {'pred._label': pred.strip()}
        else:
            return {'pred._label': pred}
    
    def parse_labels(self, preds):
        decoded = [self.parse_label(pred[0]) for pred in preds]
        
        return decoded
        
    def normalize_preds(self, actions, pred_labels):
        return pd.DataFrame({'action': actions, 'pred._label': pred_labels})


class Dexpert:
    def __init__(self, base_model_name_or_path, expert_model_name_or_path, antiexpert_model_name_or_path, task, device=use_devices):
        """
        ### Args 소개
        model_name: 불러오고자 하는 모델의 이름.
        device: GPU or CPU; GPU의 경우, 자원 최대 활용을 위하여, GPU 종류마다 배치 사이즈를 다르게 자동 구분.
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.antiexpert_model_name_or_path = antiexpert_model_name_or_path
        self.device = device
        # model
        self.model = None
        self.tokenizer = None
        self.is_lora = True
        self.task = task
        self.load_model()
        self.get_default_batch_sizes()

        self.use_tqdm = True
    
    def tqdm(self, *args, **kwargs):
        if self.use_tqdm:
            return tqdm(*args, **kwargs)
        else:
            return args[0]

    def get_default_batch_sizes(self):
        if not torch.cuda.is_available():
            self.inference_batch_size = 8
            return
        # get total memory
        # initialize total_memory
        total_memory = 0

        # iterate over all devices
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory

        # if over 80GB (a100)
        if total_memory > 80_000_000_000:
            self.inference_batch_size = 2056
        # else, if over 50GB (a6000)
        elif total_memory > 50_000_000_000:
            # self.inference_batch_size = 256
            self.inference_batch_size = 64
        else:
            # default
            self.inference_batch_size = 128
            # xxl
            # self.inference_batch_size = 32
            
    def load_model(self):
        print('Loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.expert_model_name_or_path)
        self.tokenizer.padding_side = "left"
        
        self.base = AutoModelForCausalLM.from_pretrained(self.base_model_name_or_path) # LLM
        # self.base.config.pad_token_id = self.tokenizer.pad_token_id
        # self.base.resize_token_embeddings(len(self.tokenizer)) # change embedding size
        self.antiexpert = AutoModelForCausalLM.from_pretrained(self.antiexpert_model_name_or_path) # SLM
        # self.antiexpert.config.pad_token_id = self.tokenizer.pad_token_id
        # self.antiexpert.resize_token_embeddings(len(self.tokenizer)) # change embedding size
        
        # fine-tuned SLM
        if self.is_lora:
            config = PeftConfig.from_pretrained(self.expert_model_name_or_path)
        
            self.expert = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            # self.expert.config.pad_token_id = self.tokenizer.pad_token_id

            # self.expert.resize_token_embeddings(len(self.tokenizer)) # change embedding size
            self.expert = PeftModelForCausalLM.from_pretrained(self.expert, model_id=self.expert_model_name_or_path)
        

        self.base = self.base.to(self.device)
        self.antiexpert = self.antiexpert.to(self.device)
        self.expert = self.expert.to(self.device)
        
        print("loaded FT model")
        # to device
        self.device = self.base.device

        # eval. mode
        self.base.eval()
        self.antiexpert.eval()
        self.expert.eval()
       
    def get_output(self, action):
        gens = self.get_labels(action)

        # decode
        parsed = self.parse_labels(gens)
        
        df = self.normalize_preds([act for act in action], [d['pred._label'] for d in parsed])
        return df

    def get_labels(self, actions, batch_size=None):
        # if not self.model:
        #     self.load_model() # when we evaluate computing cost, turn off.
        
        is_single = False
        if isinstance(actions, str):
            actions = [actions]
            is_single = True
        
        if batch_size is None:
            batch_size = self.inference_batch_size
        
        batch_outputs = []
        n_batchs = math.ceil(len(actions) / batch_size)
        
        for i in self.tqdm(range(n_batchs), desc="Generations"):
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            
            encoded_batch = self.tokenizer.batch_encode_plus(
                [action for action in batch_actions],
                return_tensors='pt',
                add_special_tokens=True,
                padding=True,
                truncation="longest_first",
                max_length=256
                ).to(self.device)
            
            input_ids = encoded_batch["input_ids"]
            attention_mask = encoded_batch["attention_mask"]
            
    
            with torch.no_grad():
                base_outputs = self.base.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=False, max_new_tokens=2, return_dict_in_generate=True, output_scores=True)
                
                expert_outputs = self.expert.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=False, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)
                antiexpert_outputs = self.antiexpert.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=False, max_new_tokens=2, return_dict_in_generate=True, output_scores=True)

            
            if self.task == "qasc":
                base_next_token_logits = base_outputs.scores[1]
                antiexpert_next_token_logits = antiexpert_outputs.scores[1]
            else:
                base_next_token_logits = base_outputs.scores[0]
                antiexpert_next_token_logits = antiexpert_outputs.scores[0]
            
            # # debugging code
            # bases = torch.argmax(base_next_token_logits, dim=-1)
            # base_token_ids = bases.detach().cpu().tolist()
            # base_decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False, skip_special_tokens=False) for t in base_token_ids]
            # print(base_decoded)
            # 
            
            expert_next_token_logits = expert_outputs.scores[0] # the first token
            # debugging code
            # experts = torch.argmax(expert_next_token_logits, dim=-1)
            # expert_token_ids = experts.detach().cpu().tolist()
            # expert_decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False, skip_special_tokens=False) for t in expert_token_ids]
            # print(expert_decoded)
            #
            
            
            
            # print(base_next_token_logits)
            next_token_logits = (
                base_next_token_logits +
                1 * (expert_next_token_logits - antiexpert_next_token_logits)
            )
            
            outputs = torch.argmax(next_token_logits, dim=-1)

            token_ids = outputs.detach().cpu().tolist()   

           
            decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False,
                                            skip_special_tokens=False) for t in token_ids]
            
            batch_outputs.extend(decoded)
            
        # reshape to (n_actions, n_gens)
        
        decoded = np.array(
            [t for t in batch_outputs]).reshape(len(actions), 1)
        decoded = np.array([np.array([x.strip() for x in gen]) for gen in decoded])
        
 
        # if single, return as string
        if is_single:
            decoded = decoded[0]
    
        return decoded
    
    def parse_label(self, pred):
        return {'pred._label': pred.strip()}

    def parse_labels(self, preds):
        decoded = [self.parse_label(pred[0]) for pred in preds]
        return decoded
        
    def normalize_preds(self, actions, pred_labels):
        return pd.DataFrame({'action': actions, 'pred._label': pred_labels})

class CombLM:
    def __init__(self, base_model_name_or_path, expert_model_name_or_path, is_constant, task, alpha, device=use_devices):
        """
        ### Args 소개
        model_name: 불러오고자 하는 모델의 이름.
        device: GPU or CPU; GPU의 경우, 자원 최대 활용을 위하여, GPU 종류마다 배치 사이즈를 다르게 자동 구분.
        """
        self.base_model_name_or_path = base_model_name_or_path
        self.expert_model_name_or_path = expert_model_name_or_path
        self.device = device
        self.is_constant = is_constant
        # model
        self.model = None
        self.tokenizer = None
        self.is_lora = True
        self.task = task
        self.alpha = alpha
        self.load_model()
        self.get_default_batch_sizes()

        self.use_tqdm = True
    
    def tqdm(self, *args, **kwargs):
        if self.use_tqdm:
            return tqdm(*args, **kwargs)
        else:
            return args[0]

    def get_default_batch_sizes(self):
        if not torch.cuda.is_available():
            self.inference_batch_size = 8
            return
        # get total memory
        # initialize total_memory
        total_memory = 0

        # iterate over all devices
        for i in range(torch.cuda.device_count()):
            total_memory += torch.cuda.get_device_properties(i).total_memory

        # if over 80GB (a100)
        if total_memory > 80_000_000_000:
            self.inference_batch_size = 2056
        # else, if over 50GB (a6000)
        elif total_memory > 50_000_000_000:
            # self.inference_batch_size = 256
            self.inference_batch_size = 64
        else:
            # default
            self.inference_batch_size = 128
            # xxl
            # self.inference_batch_size = 32
            
    def load_model(self):
        print('Loading model...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.expert_model_name_or_path)
        self.tokenizer.padding_side = "left"
        
        self.base = AutoModelForCausalLM.from_pretrained(self.base_model_name_or_path) # LLM
        # fine-tuned SLM
        if self.is_lora:
            config = PeftConfig.from_pretrained(self.expert_model_name_or_path)
            self.expert = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
            self.expert = PeftModelForCausalLM.from_pretrained(self.expert, model_id=self.expert_model_name_or_path)
        

        self.base = self.base.to(self.device)
        self.expert = self.expert.to(self.device)
 
        print("loaded FT model")
        # to device
        self.device = self.base.device

        # eval. mode
        self.base.eval()
        self.expert.eval()
       
    def get_output(self, action):
        gens = self.get_labels(action)

        # decode
        parsed = self.parse_labels(gens)
        
        df = self.normalize_preds([act for act in action], [d['pred._label'] for d in parsed])
        return df

    def get_labels(self, actions, batch_size=None):
        # if not self.model:
        #     self.load_model() # when we evaluate computing cost, turn off.
        
        is_single = False
        if isinstance(actions, str):
            actions = [actions]
            is_single = True
        
        if batch_size is None:
            batch_size = self.inference_batch_size
        
        batch_outputs = []
        n_batchs = math.ceil(len(actions) / batch_size)
        
        
        print(f'current alpha: {self.alpha}')
        for i in self.tqdm(range(n_batchs), desc="Generations"):
            batch_actions = actions[i*batch_size:(i+1)*batch_size]
            
            encoded_batch = self.tokenizer.batch_encode_plus(
                [action for action in batch_actions],
                return_tensors='pt',
                add_special_tokens=True,
                padding=True,
                truncation="longest_first",
                max_length=256
                ).to(self.device)
            
            input_ids = encoded_batch["input_ids"]
            attention_mask = encoded_batch["attention_mask"]
            
    
            with torch.no_grad():
                base_outputs = self.base.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=False, max_new_tokens=2, return_dict_in_generate=True, output_scores=True)
                
                expert_outputs = self.expert.generate(input_ids=input_ids, attention_mask=attention_mask,
                                                do_sample=False, max_new_tokens=1, return_dict_in_generate=True, output_scores=True)

            
            if self.task == "qasc":
                base_next_token_logits = base_outputs.scores[1] # qasc
            else:
                base_next_token_logits = base_outputs.scores[0] # others
            
            # # debugging codes
            # bases = torch.argmax(base_next_token_logits, dim=-1)
            # base_token_ids = bases.detach().cpu().tolist()
            # base_decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False, skip_special_tokens=False) for t in base_token_ids]
            # print(base_decoded)
            # 
            
            expert_next_token_logits = expert_outputs.scores[0] # the first token
            # debugging code
            # experts = torch.argmax(expert_next_token_logits, dim=-1)
            # expert_token_ids = experts.detach().cpu().tolist()
            # expert_decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False, skip_special_tokens=False) for t in expert_token_ids]
            # print(expert_decoded)
            #
            

            
            if self.is_constant: # constant scalar
                """
                alpha가 커질수록 LLM의 가중치 강도가 커진다.
                """
                next_token_logits = (
                    self.alpha * base_next_token_logits + (1-self.alpha) * expert_next_token_logits)
            else: # mean (default)
                next_token_logits = (
                base_next_token_logits +expert_next_token_logits
                )/2 # mean   
            outputs = torch.argmax(next_token_logits, dim=-1)
            # print(outputs)
            token_ids = outputs.detach().cpu().tolist()   

        
            decoded = [self.tokenizer.decode(t, clean_up_tokenization_spaces=False,
                                            skip_special_tokens=False) for t in token_ids]
            
            batch_outputs.extend(decoded)
                        
        decoded = np.array(
            [t for t in batch_outputs]).reshape(len(actions), 1)
        decoded = np.array([np.array([x.strip() for x in gen]) for gen in decoded])
        
 
        # if single, return as string
        if is_single:
            decoded = decoded[0]
    
        return decoded
    
    def parse_label(self, pred):
        return {'pred._label': pred.strip()}

    def parse_labels(self, preds):
        decoded = [self.parse_label(pred[0]) for pred in preds]
        return decoded
        
    def normalize_preds(self, actions, pred_labels):
        return pd.DataFrame({'action': actions, 'pred._label': pred_labels})


  
if __name__ == "__main__":
    base_name = "meta-llama/Llama-3.2-3B-Instruct"
    slm_name = "meta-llama/Llama-3.2-1B-Instruct"
    expert_name = "meta-llama/Llama-3.2-1B-Instruct"
    lms = Dexpert(base_name, expert_name,slm_name, task="obqa")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
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
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # standard
    dataset = ProxyDataset("stqa", "./dataset/stqa/dev.json", "./dataset/qasc/dev.jsonl", tokenizer,False)
    inputs = dataset.inputs
    labels = dataset.labels
    # 대소문자 구분 X
    decoded = lms.get_output(inputs)
    
    print(decoded["pred._label"].tolist())
