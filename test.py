# import subprocess
# import time

# # 定义一个函数来执行nvidia-smi命令
# def run_nvidia_smi():
#     try:
#         # 执行nvidia-smi命令并捕获输出
#         output = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT)
#         # 打印输出结果
#         print(output.decode('utf-8'))
#     except subprocess.CalledProcessError as e:
#         # 如果命令执行失败，打印错误信息
#         print("Failed to run nvidia-smi:", e)

# # 主循环，每600秒（即10分钟）运行一次nvidia-smi
# while True:
#     run_nvidia_smi()
#     # 等待600秒（10分钟）
#     time.sleep(60)

# # from rouge import Rouge
# # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# # from utils.prompter import Prompter

# # import torch
# # import datasets
# # from transformers import GenerationConfig
# # import os
# # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# # from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig
# # prompter = Prompter("alpaca")
# # # 假设 "/data/ty/gemma-2b" 是模型的路径
# # model_path = "/data/ty/gemma-2b"

# # # 加载tokenizer和模型
# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # model = AutoModelForCausalLM.from_pretrained(model_path).cuda() 

# # # 准备输入文本
# # input_text = "what privacy permissions are requested upon launching the app for the first time?"
# # data_point = {
# #         "instruction": "Answer the following question based on the provided context.",
# #         "input": "what privacy permissions are requested upon launching the app for the first time?",
# #         "output": "  Via Email."
# #     }
# # rouge = Rouge()
# # # 将输入文本编码成模型可以理解的格式
# # inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
# # test_prompt = prompter.generate_prompt(
# #     data_point["instruction"],
# #     data_point["input"],
# #     '### Response:',
# # )
# # target = "  Via Email."
# # ans1 = target.replace('The answer is: ','').split('.')

# # tgt_ans = None
# # for i in ans1:
# #     if len(i)>0:
# #         tgt_ans = i
# # sampling = GenerationConfig(pad_token_id = 0,
# #                                     eos_token_id = 1,
# #                                     bos_token_id = 2)
# # with torch.autocast("cuda"):
# #     inputs = tokenizer(test_prompt, return_tensors="pt")
# #     input =inputs["input_ids"].to('cuda')
# #     # if input is None:
# #     #     continue
# #     with torch.no_grad():
# #         #print(tokenizer.eos_token_id, tokenizer.pad_token_id)
# #         generation_output = model.generate(
# #             input_ids=input,
# #             generation_config=sampling,
# #             return_dict_in_generate=True,
# #             output_scores=True,
# #             max_new_tokens=40,
# #             pad_token_id=tokenizer.eos_token_id
# #         )
# #     generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
# #     # print(generation_output_decoded)
# #     # split = prompter.template["response_split"]
    
# #     split = "### Response:" 
# #     # ans = generation_output_decoded.split(split)[-1].strip()
# #     print( generation_output_decoded.split(split))
# #     ans = generation_output_decoded.split(split)[-1].strip()
# #     # if len(ans) <=0 or len(tgt_ans) <=0:
# #     #     continue
# #     print(f"ans = {ans}")
# #     print(len(ans))
# #     print(f"tgt ans = {tgt_ans}")
# #     print(len(tgt_ans))
# #     rouge_score = rouge.get_scores(ans, tgt_ans, avg=True)# 计算rouge分数
# #     print(rouge_score)


# # # # 使用模型生成文本
# # # generation_output = model.generate(**inputs, return_dict_in_generate=True,
# # #                     output_scores=True,
# # #                     max_new_tokens=100)

# # # 解码生成的序列


# # # generation_output_decoded = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)

# # # for i in ans1:
# # #     print(len(i))
# # # print(ans1)
# # # print(tgt_ans)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from datasets import load_dataset
model_name = "/data/LLM_models/llama-7b"  
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
local_data_path="/data/FL-DD classification dataset/iid_dataset/mix1.json"
train_dataset=load_dataset("json", data_files=local_data_path)
tokenized_dataset = tokenize_function(train_dataset)
training_args = TrainingArguments(
output_dir='./results',          # 输出目录的路径。
num_train_epochs=3,              # 训练轮数。
per_device_train_batch_size=16,  # 每个设备上的批大小。
per_device_eval_batch_size=64,   # 每个设备上的评估批大小。
warmup_steps=500,                # 预热步数。
weight_decay=0.01,               # 权重衰减。
logging_dir='./logs',            # 日志目录的路径。
)
print(tokenized_dataset)
trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

trainer.train()
eval_dataset = "/data/FL-DD classification dataset/iid_dataset/mix2.py"
eval_results = trainer.evaluate(eval_dataset)
print(eval_results)
# inputs = tokenizer("This is a sample text for classification.", return_tensors="pt")
# outputs = model(**inputs)
# logits = outputs.logits


# probabilities = torch.nn.functional.softmax(logits, dim=-1)
# predicted_class = torch.argmax(probabilities, dim=-1)
# print(f"Predicted class: {predicted_class.item()}")