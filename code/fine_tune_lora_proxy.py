from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from huggingface_hub import login
import argparse
import torch
from dataset_proxy import ProxyDataset
from peft import LoraConfig, get_peft_model, TaskType
import utils
# for utilizing GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#### argument #### 
parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--task', help='piqa, csqa, stqa', type=str, required=True)
parser.add_argument('--seed', help='seed (2, 10, or 42)', type=int, required=True)

# required = False
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=False, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--per_device_train_batch_size', help='total batch size / # of gradient accumulation steps', type=int, required=False, default=16)
parser.add_argument('--gradient_accumulation_steps', help='# of gradient accumulation steps', type=int, required=False, default=1)
parser.add_argument('--save_path', help='path where the aligner model ckpt to be saved', type=str, required=False, default='./models')
parser.add_argument('--logging_dir', help='path where the logging of aligner model to be saved', type=str, required=False, default="./runs")
parser.add_argument('--lr', help='learning rate', type=float, required=False, default=2e-4)
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler', type=str, required=False, default="cosine")
parser.add_argument('--epoch', help='training epoch', type=int, required=False, default=6)
# parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0.03)
parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0)
parser.add_argument('--weight_decay', help='weight decay', type=float, required=False, default=0.0)
parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False, default="your huggingface key")
args = parser.parse_args()



task = args.task
seed = args.seed
model_ckpt = args.model_ckpt
model_name = model_ckpt.split("/")[1]
lr = args.lr
lr_scheduler_type = args.lr_scheduler_type
epoch = args.epoch
per_device_train_batch_size = args.per_device_train_batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
lr_warmup_ratio = args.lr_warmup_ratio
weight_decay = args.weight_decay
is_lora = True
    

if task == "piqa":
    question_path = "./dataset/piqa/train.jsonl"
    answer_path = "./dataset/piqa/train-labels.lst"
elif task == "csqa":
    question_path = "./dataset/csqa/train_rand_split.jsonl"
    answer_path = "./dataset/csqa/train_rand_split.jsonl"
elif task == "qasc":
    question_path = "./dataset/qasc/train.jsonl"
    answer_path = "./dataset/qasc/train.jsonl"
elif task in ["arc-e", "arc-h", "obqa"]:
    question_path = None
    answer_path = None
elif task in ["wngr", "boolq"]:
    question_path = None
    answer_path = None
elif task == "siqa":
    question_path = None
    answer_path = None
elif task == "stqa":
    question_path = "./dataset/stqa/train.json"
    answer_path = "./dataset/stqa/train.json"


save_path = f"{args.save_path}/proxy/{task}/cls/{model_name}/lora_{is_lora}/{seed}"
logging_dir = f"{args.logging_dir}/proxy/{task}/cls/{model_name}/lora_{is_lora}/{seed}"

##############################################################
################# main code ##################################
##############################################################
api_key = args.huggingface_api_key
login(token=api_key)

# 어떠한 스페셜 토큰을 추가하지 않는다.
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right" # standard
### main model ###
base_model = AutoModelForCausalLM.from_pretrained(model_ckpt)
base_model.config.pad_token_id = tokenizer.eos_token_id
if is_lora:
    lora_config = LoraConfig(
        r=64,
        lora_alpha=8,
        target_modules=utils.target_module(base_model), # all linear
        lora_dropout=0.1,
        bias="none",
        inference_mode=False, 
        task_type=TaskType.CAUSAL_LM,
    )

    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()
print(f"### loaded PLM ###")

dataset = ProxyDataset(task, question_path, answer_path, tokenizer, True)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False, pad_to_multiple_of=8)
print("### loaded dataset ###")

training_args = TrainingArguments(
    output_dir=save_path,
    logging_strategy='steps',
    logging_steps=5,
    torch_compile=True,
    save_strategy="epoch",
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
    gradient_checkpointing=False
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