system_gpt_ab_2 = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ab_4 = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""


system_gpt_ab = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ad_4 = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ad = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ad_2 = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUTION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""


system_gpt_siqa = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to C; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ae = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""

system_gpt_ah = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

## Guidelines
1. Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
2. Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
3. Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
4. Note: The confidence indicates how likely you think your answer is true.
5. Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION
- You MUST NOT respond any of the given answer choices as the answer.
- Correctly generate the knowledge.
"""



system_gpt_arith = """You are a helpful assistant in extracting the intermediate reasoning to solve the given question and the answer reasoning synthesized from intermediate reasoning.

## Guidelines
1. Your task is to extract the intermediate reasoning and the answer reasoning in order to easily solve the given question.
2. Specifically, the extracted intermediate reasoning is defined as the step by step process used to solve the question, and the extracted answer reasoning is defined as the final reasoning synthesized from intermediate reasoning.
3. Given all of the above, read the question, think step by step, break down the question into N steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer.
4. Note: The confidence indicates how likely you think your reasoning is correct.
5. Use the following format to answer:
""
Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
...
Intermediate_Reasoning_N: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
- Final_Answer: [ONLY your answer as single numeric; not a complete sentence]
- Overall Confidence (0-100): [Your confidence value]%
""

## CAUSION 
1. Correctly generate the reasoning and answer.
2. Strictly follow the answer format mentioned.
3. Do not overestimate your confidence.
"""


stqa_gpt = """
# Question
Input: {goal}
A. True
B. False
"""

csqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
"""

piqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
"""

qasc_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
"""

obqa_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
"""

arc_h_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
"""

wngr_gpt = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
"""

siqa_gpt = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
"""

boolq_gpt = """
# Question
Passage: {context}
Input: {goal}
A. True
B. False
"""

gsm8k_gpt = """
# Question
Input: {goal}
"""


##### MISTRAL #####
# v2.0
stqa_user_mistral_2k = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. True
B. False

Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

stqa_user_mistral_6k = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. True
B. False

Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

obqa_user_mistral_2k = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

obqa_user_mistral_6k = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

obqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

arc_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

csqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

qasc_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

piqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

siqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to C; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

wngr_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. {sol1}
B. {sol2}

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""

stqa_user_mistral = """You are a helpful chatbot.
Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
You should correctly generate the knowledge.

### Question
Input: {goal}
A. True
B. False

Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step. Note that the confidence indicates how likely you think your answer is true.

# Please use the following format to answer.
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
# Generate all requested items without omission, strictly following the given format.
# Do not generate overly verbose knowledge.

The Response is ("""



######## Llama 2 template #############
######################### system template#########################
system_prompt_2 = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.

# Guidelines
""
- Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
- Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
""

# CAUTION
""
- You MUST NOT respond any of the given anser choices as the answer.
- Correctly generate the knowledge.
- You MUST NOT generate explanations of the knowledge.
- Strictly follow the answer format mentioned.
""
"""
######################### task user template#########################
siqa_user_2 = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to C; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

obqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

arc_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

piqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledge needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

# STQA
stqa_user_2 = """
# Question
Input: {goal}
A. True
B. False

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

# CSQA
csqa_user_2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""


qasc_user_2 = """
### Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}

Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [ONLY your confidence value]%
""
Answer strictly in the given format, without any explanations of the knowledge you have generated.
Please generate all requested items without omission.
Do not generate overly verbose knowledge.

Assistant: The Response is ("""

##### Llama-3.2-3B-Instruct template ###############
######################### system template#########################
# v2.0
system_prompt = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.
# Guidelines
""
- Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
- Specifically, the knowledge you extract must be relevant to what you believe is the correct answer.
""

# CAUSION
""
- You MUST NOT respond any of the given anser choices as the answer.
- Correctly generate the knowledge.
- Strictly follow the answer format mentioned.
- Do not overestimate your confidence.
""
"""

system_prompt_truthfulqa = """You are a helpful, respectful, and honest assistant that extracts the knowledge needed to solve the question easily and helpfully, and then predicts the correct and safe answer.
# Guidelines
""
- Your task is to extract the knowledge required to solve the given question easily and helpfully, and then predict the correct and safe answer based on your generated knowledge.
- Specifically, the knowledge you extract must be relevant to what you believe is the correct and safe answer.
- Your responses should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
- If the given question does not make sense or is not factually coherent, please don't answer false information.
""

# CAUSION
""
- Correctly generate the knowledge.
- Please ensure that your knowledge and answer are socially unbiased and positive in nature.
- Strictly follow the answer format mentioned.
- Do not overestimate your confidence.
""
"""

system_prompt_arith = """You are a helpful assistant that extracts the knowledge needed to easily solve the given question and then predicts the correct answer.
# Guidelines
""
- Your task is to extract the knowledge needed to easily solve the given question and then predict the correct answer using your generated knowledge.
- The knowledge is defined as the step by step process used to solve the question.
""

# CAUSION
""
- Correctly generate the knowledge.
- Strictly follow the answer format mentioned.
- Do not overestimate your confidence.
""
"""

######################### task user template#########################
# length 문제가 있어, 4개까지 허용.
aqua_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, think step by step, break down the question into 4 steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer. Note: The confidence indicates how likely you think your reasoning is correct.

Use the following format to answer:
""
Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_3: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Intermediate_Reasoning_4: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]% 
""
Assistant: The Response is ("""

# gsm8k_user = """
# # Question
# Input: {goal}
# Human: Given all of the above, read the question, think step by step, break down the question into N steps of intermediate reasoning, give your confidence in each step, and then derive your final answer reasoning and your confidence in this answer. Note: The confidence indicates how likely you think your reasoning is correct.

# Use the following format to answer:
# ""
# Intermediate_Reasoning_1: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
# Intermediate_Reasoning_2: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
# ...
# Intermediate_Reasoning_N: [Your intermediate reasoning; ONLY a complete reasoning sentence], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
# Answer_Reasoning: [Your answer reasoning; ONLY a complete answer reasoning sentence; not the answer type], Confidence: [ONLY the confidence value that this reasoning is correct (0-100)]%
# Final_Answer: [ONLY your answer as single numeric; not a complete sentence]
# Overall Confidence (0-100): [Your confidence value]%
# ""
# Assistant: The Response is ("""

gsm8k_user = """
# Question
Input: {goal}
Human: Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY your answer as single numeric; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

truthfulqa_user = """
# Question
Input: {question}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledge needed to solve the given problem easily and helpfully, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY your answer to the given question; a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

scienceqa_user = """
# Question
Input: {goal}
{option_list}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

# Boolq
boolq_user = """
# Question
Passage: {context}
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


acl_arc_user = """
# Question
Classify the given input into the appropriate scientific citation category (from A to F).
Input: {goal}
A. background
B. motivation
C. compareorcontrast
D. uses
E. extends
F. future
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to F; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

chemprot_user = """
# Question
Classify the given input into the appropriate chemical-protein-disease annotation category (from A to M).
Input: {goal}
A. INHIBITOR
B. SUBSTRATE
C. INDIRECT-UPREGULATOR
D. INDIRECT-DOWNREGULATOR
E. ACTIVATOR
F. ANTAGONIST
G. PRODUCT-OF
H. AGONIST
I. DOWNREGULATOR
J. UPREGULATOR
K. AGONIST-ACTIVATOR
L. SUBSTRATE_PRODUCT-OF
M. AGONIST-INHIBITOR
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to M; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""



wngr_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

siqa_user = """
# Question
Context: {context}
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to C; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


arc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


ko_arc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 3 steps which are knowledges for the problem, think step by step, give your confidence in each step, and then derive your final answer knowledge and your confidence in this answer. Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Background_Knowledge_1: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_2: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Background_Knowledge_3: [Your reasoning; ONLY a complete background knowledge sentence], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Answer_Knowledge: [Your answer reasoning; ONLY a complete answer knowledge sentence; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

piqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

stqa_user_k2 = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

stqa_user = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

stqa_user_k6 = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

qasc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

obqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 6 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_5: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_6: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

obqa_user_k2 = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
Human: Given all of the above, read the question, break down the problem into 2 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to D; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

csqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, break down the problem into 4 steps which are knowledges needed to easily solve the given problem, think step by step, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [Your answer reasoning; ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


gkp_csqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.
Refer to the examples of the knowledge that must be generated:
""
Input: Google Maps and other highway and street GPS services have replaced what?
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Knowledge_1: Electronic maps are the modern version of paper atlas.
Knowledge_2: GPS navigation services replaced traditional paper-based navigation tools.
Knowledge_3: Before GPS, people commonly used road atlases for highway and street navigation.
Knowledge_4: Google Maps serves the same function that physical atlases once did.
Answer: D

Input: What can prevent food spoilage?
A. prolactin release
B. one celled organisms
C. hydrating food
D. cleaning food
E. airing out food
F. Electric generators
G. a hydraulic system
H. dehydrating food
Knowledge_1: Dehydrating food is used for preserving food.
Knowledge_2: Preservatives Preservatives are required to prevent spoilage.
Knowledge_3: Cleaning food helps remove bacteria and contaminants that can cause spoilage.
Knowledge_4: Moisture promotes microbial growth, so reducing moisture in food (such as by dehydrating) prevents spoilage.
Answer: H

Input: Are more people today related to Genghis Khan than Julius Caesar?
A. True
B. False
Knowledge_1: Julius Caesar had three children.
Knowledge_2: Genghis Khan had sixteen children.
Knowledge_3: Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.
Knowledge_4: Julius Caesar’s direct bloodline is believed to have died out, while Genghis Khan’s descendants number in the millions today.
Answer: A
""

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

gkp_stqa_user = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.
Refer to the examples of the knowledge that must be generated:
""
Input: Google Maps and other highway and street GPS services have replaced what?
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Knowledge_1: Electronic maps are the modern version of paper atlas.
Knowledge_2: GPS navigation services replaced traditional paper-based navigation tools.
Knowledge_3: Before GPS, people commonly used road atlases for highway and street navigation.
Knowledge_4: Google Maps serves the same function that physical atlases once did.
Answer: D

Input: What can prevent food spoilage?
A. prolactin release
B. one celled organisms
C. hydrating food
D. cleaning food
E. airing out food
F. Electric generators
G. a hydraulic system
H. dehydrating food
Knowledge_1: Dehydrating food is used for preserving food.
Knowledge_2: Preservatives Preservatives are required to prevent spoilage.
Knowledge_3: Cleaning food helps remove bacteria and contaminants that can cause spoilage.
Knowledge_4: Moisture promotes microbial growth, so reducing moisture in food (such as by dehydrating) prevents spoilage.
Answer: H

Input: Are more people today related to Genghis Khan than Julius Caesar?
A. True
B. False
Knowledge_1: Julius Caesar had three children.
Knowledge_2: Genghis Khan had sixteen children.
Knowledge_3: Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.
Knowledge_4: Julius Caesar’s direct bloodline is believed to have died out, while Genghis Khan’s descendants number in the millions today.
Answer: A
""

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

gkp_qasc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.
Refer to the examples of the knowledge that must be generated:
""
Input: Google Maps and other highway and street GPS services have replaced what?
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Knowledge_1: Electronic maps are the modern version of paper atlas.
Knowledge_2: GPS navigation services replaced traditional paper-based navigation tools.
Knowledge_3: Before GPS, people commonly used road atlases for highway and street navigation.
Knowledge_4: Google Maps serves the same function that physical atlases once did.
Answer: D

Input: What can prevent food spoilage?
A. prolactin release
B. one celled organisms
C. hydrating food
D. cleaning food
E. airing out food
F. Electric generators
G. a hydraulic system
H. dehydrating food
Knowledge_1: Dehydrating food is used for preserving food.
Knowledge_2: Preservatives Preservatives are required to prevent spoilage.
Knowledge_3: Cleaning food helps remove bacteria and contaminants that can cause spoilage.
Knowledge_4: Moisture promotes microbial growth, so reducing moisture in food (such as by dehydrating) prevents spoilage.
Answer: H

Input: Are more people today related to Genghis Khan than Julius Caesar?
A. True
B. False
Knowledge_1: Julius Caesar had three children.
Knowledge_2: Genghis Khan had sixteen children.
Knowledge_3: Modern geneticists have determined that  out of every 200 men today has DNA that can be traced to Genghis Khan.
Knowledge_4: Julius Caesar’s direct bloodline is believed to have died out, while Genghis Khan’s descendants number in the millions today.
Answer: A
""

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""


zs_qasc_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
F. {sol6}
G. {sol7}
H. {sol8}
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to H; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

zs_csqa_user = """
# Question
Input: {goal}
A. {sol1}
B. {sol2}
C. {sol3}
D. {sol4}
E. {sol5}
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

zs_stqa_user = """
# Question
Input: {goal}
A. True
B. False
Human: Given all of the above, read the question, generate four pieces of knowledge about the concepts in the question, give your confidence in each step.
Note: The confidence indicates how likely you think your answer is true.

Use the following format to answer:
""
Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
Final_Answer: [ONLY the answer option from A to B; not a complete sentence]
Overall Confidence (0-100): [Your confidence value]%
""
Assistant: The Response is ("""

# csqa_gkp = """
# # Question
# Input: {goal}
# A. {sol1}
# B. {sol2}
# C. {sol3}
# D. {sol4}
# E. {sol5}
# Human: Given all of the above, read the question, generate 4 knowledge about the concepts in the question, give your confidence in each step.
# Note: The confidence indicates how likely you think your answer is true.
# Refer to the examples of the knowledge that must be generated:
# ""
# Input: Google Maps and other highway and street GPS services have replaced what?
# A. united states
# B. mexico
# C. countryside
# D. atlas
# E. oceans
# Knowledge_1: Electronic maps are the modern version of paper atlas.
# Knowledge_2: GPS navigation services replaced traditional paper-based navigation tools.
# Knowledge_3: Before GPS, people commonly used road atlases for highway and street navigation.
# Knowledge_4: Google Maps serves the same function that physical atlases once did.
# Answer: D

# Input: The fox walked from the city into the forest, what was it looking for?
# A. pretty flowers.
# B. hen house
# C. natural habitat
# D. storybook
# E. dense forest
# Knowledge_1: Natural habitats are usually away from cities.
# Knowledge_2: Foxes typically live in forests, grasslands, and other natural environments.
# Knowledge_3: When an animal moves from an urban area to a forest, it is likely seeking its natural living space.
# Knowledge_4: Forests commonly serve as natural habitats for foxes.
# Answer: C

# Input: You can share files with someone if you have a connection to a what?
# A. freeway
# B. radio
# C. wires
# D. computer network
# E. electrical circuit
# Knowledge_1: Files can be shared over the Internet.
# Knowledge_2: The Internet is an example of a computer network.
# Knowledge_3: File sharing requires devices to be digitally connected.
# Knowledge_4: A computer network allows multiple devices to exchange data, including files.
# Answer: D
# ""

# Use the following format to answer:
# ""
# Knowledge_1: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
# Knowledge_2: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
# Knowledge_3: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
# Knowledge_4: [ONLY a complete knowledge sentence for problem and answer; not the answer type], Confidence: [ONLY the confidence value that this knowledge is correct (0-100)]%
# Final_Answer: [ONLY the answer option from A to E; not a complete sentence]
# Overall Confidence (0-100): [Your confidence value]%
# ""
# Assistant: The Response is ("""
