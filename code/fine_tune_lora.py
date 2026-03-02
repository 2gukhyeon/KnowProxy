from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from huggingface_hub import login
import argparse
import torch
from dataset_gen import Gen_Dataset, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import utils
# for utilizing GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#### argument #### 
parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=True, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--task', help='piqa, csqa, stqa', type=str, required=True)
parser.add_argument('--back_ratio', help='background knowledge ratio: 0~1', type=float, required=True)
parser.add_argument('--is_ours', help='alinger variant', type=str, required=True)
parser.add_argument('--seed', help='seed (2, 10, or 42)', type=int, required=True)
parser.add_argument('--llm', help='llama3.2, gpt3.5, llama2, mistral', type=str, required=True)


# gsm8k experimental setting: 1epoch, 1batch
# otherwise: 6 epoch, 16 batch
# required = False
parser.add_argument('--ratio', help='reasoning thres.', type=float, required=False, default=-1)
parser.add_argument('--is_lora', help='Llama-3.2-3B-Instruct', type=str, required=False, default="yes")
parser.add_argument('--per_device_train_batch_size', help='total batch size / # of gradient accumulation steps', type=int, required=False, default=4)
parser.add_argument('--gradient_accumulation_steps', help='# of gradient accumulation steps', type=int, required=False, default=2)
parser.add_argument('--save_path', help='path where the aligner model ckpt to be saved', type=str, required=False, default='./models')
parser.add_argument('--logging_dir', help='path where the logging of aligner model to be saved', type=str, required=False, default="./runs")
parser.add_argument('--lr', help='learning rate', type=float, required=False, default=1e-5) # 2e-4 default, 2e-5(GSM8K: 71.34192570128886), 1e-5, 3epoch (100step: GSM8K: 77.5587566338135 )
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler', type=str, required=False, default="cosine")
parser.add_argument('--epoch', help='training epoch', type=int, required=False, default=6)
# parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0.03)
parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0.05)
parser.add_argument('--weight_decay', help='weight decay', type=float, required=False, default=0.01)
parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False, default="your huggingface key")
args = parser.parse_args()


ratio = args.ratio
back_ratio = args.back_ratio # for training; quality up training knowledge
task = args.task
seed = args.seed
llm = args.llm
model_ckpt = args.model_ckpt
model_name = model_ckpt.split("/")[1]
lr = args.lr
lr_scheduler_type = args.lr_scheduler_type
epoch = args.epoch
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
lr_warmup_ratio = args.lr_warmup_ratio
weight_decay = args.weight_decay

if args.is_ours == "yes":
    is_ours = True
else:
    is_ours = False
    
if args.is_lora == "yes":
    is_lora = True
else:
    is_lora = False
    

if task in ["gsm8k", "aqua"]:
    question_path = None
    answer_path = None
elif task == "truthfulqa":
    question_path = "./dataset/TruthfulQA/data/v1/mc_task_train_gen.json"
    answer_path = None
elif task == "scienceqa":
    question_path = "./dataset/scienceqa/scienceqa_train.json"
    answer_path = None

    

cq_path = f"./dataset/{task}/{llm}/converted_train.json"  
save_path = f"{args.save_path}/{llm}/{task}/gen/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"
logging_dir = f"{args.logging_dir}/{llm}/{task}/gen/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"

##############################################################
################# main code ##################################
##############################################################
api_key = args.huggingface_api_key
login(token=api_key)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# eos랑 pad를 같게 해선 안된다. eos에 대한 로스가 사라져, 문장이 그냥 길어짐.
special_tokens_dict = {}
special_token = []
if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_tokenizeren_id = tokenizer.eos_token_id
    special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    special_token.append(DEFAULT_PAD_TOKEN)
    print("add pad token")
if tokenizer.eos_token is None:
    special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
    special_token.append(DEFAULT_EOS_TOKEN)
    print("add eos token")
if tokenizer.bos_token is None:
    special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
    special_token.append(DEFAULT_BOS_TOKEN)
    print("add bos token")
    None
if tokenizer.unk_token is None:
    # tokenizer.unk_token = tokenizer.eos_token
    # tokenizer.unk_token_id = tokenizer.eos_token_id
    special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN
    special_token.append(DEFAULT_UNK_TOKEN)
    print("add unk token")
    
# token_list = ['[INST]', "[/INST]"] # 나중에 풀기 
# special_tokens_dict['additional_special_tokens'] = token_list # 나중에 풀기
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.padding_side = "right" # generative methods

# model_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type="nf4",
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=torch.bfloat16
#         )


### main model ###
base_model = AutoModelForCausalLM.from_pretrained(model_ckpt,
                                                #   quantization_config=model_config,
                                                  torch_dtype=torch.bfloat16
                                                  )
base_model.resize_token_embeddings(len(tokenizer))
base_model.config.pad_token_id = tokenizer.pad_token_id
# base_model = prepare_model_for_kbit_training(base_model)

if is_lora:
    lora_config = LoraConfig(
        r=32,
        lora_alpha=32,
        # target_modules=utils.target_module(base_model), # all linear
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
        lora_dropout=0.1,
        bias="none",
        inference_mode=False, 
        task_type=TaskType.CAUSAL_LM,
        trainable_token_indices={'embed_tokens': tokenizer.convert_tokens_to_ids(special_token)},
    )

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()
    # base_model = base_model.to(torch.bfloat16)
print(f"### loaded PLM ###")


dataset = Gen_Dataset(task, question_path, answer_path, tokenizer, True, is_ours, cq_path, ratio, back_ratio)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
print("### loaded dataset ###")

training_args = TrainingArguments(
    output_dir=save_path,
    logging_strategy='steps',
    logging_steps=5,
    torch_compile=True,
    save_strategy="steps",
    save_steps=100,
    # save_strategy="epoch",
    num_train_epochs=epoch,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=lr_warmup_ratio,
    weight_decay=weight_decay,
    seed=42,
    report_to='tensorboard',
    logging_dir=logging_dir,
    gradient_checkpointing=False,
    # bf16=True,
    # fp16=True
)

trainer = Trainer(
    model=base_model.to(device),
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print('### start fine-tuning ###')
base_model.config.use_cache=False
trainer.train()
print('### ended fine-tuning ###')