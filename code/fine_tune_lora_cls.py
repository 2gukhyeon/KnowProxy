from transformers import DataCollatorWithPadding, DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, LlamaForSequenceClassification, LlamaConfig, T5ForConditionalGeneration
from huggingface_hub import login
import argparse
import torch
from dataset import CorrectionJSONDataset, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

from peft import LoraConfig, get_peft_model, TaskType
import utils
from baseline_dataset import BaselineDataset
# for utilizing GPU
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

#### argument #### 
parser = argparse.ArgumentParser()
# required = True
parser.add_argument('--model_ckpt', help='pre-trained open-source model: huggingface_model_path', type=str, required=True, default="google/gemma-2b")
parser.add_argument('--task', help='piqa, csqa, stqa', type=str, required=True)
parser.add_argument('--back_ratio', help='background knowledge ratio: 0~1', type=float, required=True)
parser.add_argument('--is_ours', help='alinger variant', type=str, required=True)
parser.add_argument('--seed', help='seed (2, 10, or 42)', type=int, required=True)
parser.add_argument('--llm', help='llama3.2, gpt3.5, llama2, mistral', type=str, required=True)

# required = False
parser.add_argument('--ratio', help='reasoning thres.', type=float, required=False, default=-1)
parser.add_argument('--is_seq2seq', help='using seq2seq', type=str, required=False, default="no")
parser.add_argument('--per_device_train_batch_size', help='total batch size / # of gradient accumulation steps', type=int, required=False, default=16)
parser.add_argument('--gradient_accumulation_steps', help='# of gradient accumulation steps', type=int, required=False, default=1)
parser.add_argument('--save_path', help='path where the aligner model ckpt to be saved', type=str, required=False, default='./models')
parser.add_argument('--logging_dir', help='path where the logging of aligner model to be saved', type=str, required=False, default="./runs")
parser.add_argument('--lr', help='learning rate', type=float, required=False, default=2e-4)
parser.add_argument('--lr_scheduler_type', help='learning rate scheduler', type=str, required=False, default="cosine")
parser.add_argument('--epoch', help='training epoch', type=int, required=False, default=6)
parser.add_argument('--lr_warmup_ratio', help='warmup step ratio, which is # of steps ("total steps * ratio")', type=float, required=False, default=0) # llama
parser.add_argument('--weight_decay', help='weight decay', type=float, required=False, default=0.0)
parser.add_argument('--huggingface_api_key', help='huggingface api key for gemma, llama ...', type=str, required=False, default="your huggingface key")
parser.add_argument('--is_lora', help='use LoRA', type=str, required=False, default='yes')
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

if args.is_seq2seq == "yes":
    is_seq2seq = True
else:
    is_seq2seq = False

if task == "piqa":
    num_labels = 2
    question_path = "./dataset/piqa/train.jsonl"
    answer_path = "./dataset/piqa/train-labels.lst"  
elif task == "csqa":
    num_labels = 5
    question_path = "./dataset/csqa/train_rand_split.jsonl"
    answer_path = "./dataset/csqa/train_rand_split.jsonl"
elif task == "mcsqa_ru":
    num_labels = 5
    question_path = None
    answer_path = None
elif task == "mcsqa_fr":
    num_labels = 5
    question_path = None
    answer_path = None
elif task == "stqa":
    num_labels = 2
    question_path = "./dataset/stqa/train.json"
    answer_path = "./dataset/stqa/train.json"
elif task == "qasc":
    num_labels = 8
    question_path = "./dataset/qasc/train.jsonl"
    answer_path = "./dataset/qasc/train.jsonl"
elif task in ["arc-e", "arc-h", "obqa", "ko-arc"]:
    num_labels = 4
    question_path = None
    answer_path = None
elif task in ["wngr", "boolq"]:
    num_labels = 2
    question_path = None
    answer_path = None
elif task == "siqa":
    num_labels = 3
    question_path = None
    answer_path = None
elif task == "chemprot":
    num_labels = 13
    question_path = "./dataset/domain_classification/chemprot/train.txt"
    answer_path = None
elif task == "acl_arc":
    num_labels = 6
    question_path = "./dataset/domain_classification/citation_intent/train.txt"
    answer_path = None


cq_path = f"./dataset/{task}/{llm}/converted_train.json"
save_path = f"{args.save_path}/{llm}/{task}/cls/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"
logging_dir = f"{args.logging_dir}/{llm}/{task}/cls/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"
# logging_dir = f"{args.logging_dir}/{llm}/{task}/2k/cls/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"
# logging_dir = f"{args.logging_dir}/{llm}/{task}/6k/cls/{model_name}/{is_ours}/lora_{is_lora}/{seed}/{back_ratio}"

##############################################################
################# main code ##################################
##############################################################
api_key = args.huggingface_api_key
login(token=api_key)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
# eos랑 pad를 같게 해선 안된다. eos에 대한 로스가 사라져, 문장이 그냥 길어짐.
special_tokens_dict = {}
if tokenizer.pad_token is None:
    special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    # tokenizer.pad_token = tokenizer.eos_token
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
tokenizer.padding_side = "right" # standard methods

### main model ###
if is_seq2seq:
    base_model = T5ForConditionalGeneration.from_pretrained(model_ckpt)
    base_model.resize_token_embeddings(len(tokenizer))
else:
    base_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=num_labels)
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id

if is_lora:
    if is_seq2seq:
        target_modules = utils.target_module(base_model) 
        print(f'target_module for lora: {target_modules}')
        lora_config = LoraConfig(
            r=64,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules = target_modules,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        print(f"### loaded seq2seq w/ lora model ###")
    else:
        lora_config = LoraConfig(
        r=64,
        lora_alpha=8,
        target_modules='all-linear',
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
        )
        print(f"### loaded generation w/ lora model ###")
    base_model = get_peft_model(base_model, lora_config)
    base_model.print_trainable_parameters()



if is_seq2seq:
    dataset = Seq2SeqDataset(task, question_path, answer_path, tokenizer, True, is_ours, cq_path)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=base_model, pad_to_multiple_of=8)
else:
    dataset = CorrectionJSONDataset(task, question_path, answer_path, tokenizer, True, is_ours, cq_path, ratio, back_ratio) # OURS
    # dataset = BaselineDataset(task, question_path, answer_path, tokenizer,True, None, "gkp") # baseline
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
print("### loaded dataset ###")

training_args = TrainingArguments(
    output_dir=save_path,
    logging_strategy='steps',
    logging_steps=50,
    torch_compile=True,
    save_strategy="epoch",
    num_train_epochs=epoch,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=lr,
    lr_scheduler_type=lr_scheduler_type,
    warmup_ratio=lr_warmup_ratio,
    weight_decay=weight_decay,
    seed=seed,
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