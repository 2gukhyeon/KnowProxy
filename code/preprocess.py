
# PROMPT_BEGIN: str = '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# Cutting Knowledge Date: December 2023
# Today Date: 25 Sep 2025

# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>'''.strip()
# PROMPT_USER: str = '''\n\n{input}<|eot_id|>''' # llama3
# PROMPT_ASSISTANT: str = '<|start_header_id|>assistant<|end_header_id|>\n\n'  # llama3

# PROMPT_USER: str = '<s>[INST] {input} ' # llama2-chat


PROMPT_BEGIN: str = '' # classification
PROMPT_USER: str = '[INST] {input} ' # classification
PROMPT_ASSISTANT: str = '[/INST]'  # classification and llama2-chat




PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

def preprocess(input, eos_token):
    data = [PROMPT_BEGIN]
    for idx, line in enumerate([input]):
        # PROMPT_INPUT
        if idx % 2 == 0:
            data.extend((PROMPT_USER.format(input=line), PROMPT_ASSISTANT))
        else:
            if eos_token == "empty": # inference
                None
            else: # train
                data.extend((line, eos_token))

    return ''.join(data)


# def chattemplate(input, output):
#     messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": input.strip()},
#     {"role": "assistant", "content": output.strip()}
#     ]
    
#     return
    
    
       