import os
from typing import List
from tqdm import tqdm
import fire
import torch
import datasets
from transformers import GenerationConfig
import json
import math
from collections import Counter
from peft import set_peft_model_state_dict
import numpy as np
import random
from rouge import Rouge
# from nltk.translate.bleu_score import sentence_bleu
# from nltk.tokenize import word_tokenize

model_type = 'llama'
datasets.utils.logging.set_verbosity_error()
device_map = "auto"
max_new_token: int = 32
verbose: bool = False

# 设置随机数种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)

def ngram_precision(candidate, reference, n):
    # 提取候选字符串和参考字符串的n-grams
    candidate_ngrams = Counter([counter for counter in zip(*[candidate[i:] for i in range(n)])])
    reference_ngrams = Counter([counter for counter in zip(*[reference[i:] for i in range(n)])])
    clipped_count = sum(min(candidate_ngrams[gram], reference_ngrams[gram]) for gram in candidate_ngrams)
    all_count = sum(candidate_ngrams[gram] for gram in candidate_ngrams)
    precision = clipped_count / all_count if candidate_ngrams else 0
    return precision

def brevity_penalty(candidate, reference):
    if len(candidate) > len(reference):
        return 1
    ratio = len(candidate) / len(reference) if len(reference) > 0 else 0
    return math.exp(1 - ratio) if ratio < 1 else 1

def sentence_bleu(candidate, reference, max_n=4):
    # 计算BLEU分数
    p_ns = [ngram_precision(candidate, reference, n) for n in range(1, max_n + 1) ]
    p_ns = [p for p in p_ns if p > 0]  # 移除0值
    if not p_ns:
        return 0  # 如果没有匹配的n-grams，则BLEU分数为0

    geo_mean = math.exp(math.fsum(math.log(p) for p in p_ns) / len(p_ns))
    bp = brevity_penalty(candidate, reference) 
    return bp * geo_mean

def global_evaluation(model, tokenizer, prompter, dev_data_path):
    # data_class =  ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science', 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
    # right_count_dict = dict.fromkeys(data_class, 0)
    # total_count_dict = dict.fromkeys(data_class, 0)
    # acc_count_dict = dict.fromkeys(data_class, 0)
    # with open(dev_data_path, 'r') as f:
    #     test_set = json.load(f)
    

    test_set = []  
    for file in dev_data_path:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_set.extend(data)
        count=0

    if model_type == 'llama':
        sampling = GenerationConfig(
            do_sample=True,
            temperature=0.2,
            top_p=0.6,
            top_k=30,
            num_beams=1,
            max_new_tokens=max_new_token,
            early_stopping=False,
        )

    if model_type == 'gpt2':
        sampling = GenerationConfig(
            bos_token_id = 50256,
            eos_token_id = 50256,
            _from_model_config = True,
        )

    rouge = Rouge()
    score_rouge = []
    score_bleu = []
    for data_point in tqdm(test_set):
        count +=1
        target = data_point["output"]
        # class_test_set = data_point["class"]
        
        tgt_ans_idx = target.replace('The answer is: ','').split('. ')
        # print(tgt_ans_idx)
        # tgt_ans = target.replace('The answer is: ','').split('. ')[1]
        if len(tgt_ans_idx)>1:
            tgt_ans = tgt_ans_idx[1]
        else:
            tgt_ans = tgt_ans_idx[0]

        if len(data_point["input"])==0:
            continue

        test_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            'The answer is: ',
        )

        with torch.autocast("cuda"):
            inputs = tokenizer(test_prompt, return_tensors="pt")
            input =inputs["input_ids"].to('cuda')
            if input is None:
                continue
            with torch.no_grad():
                #print(tokenizer.eos_token_id, tokenizer.pad_token_id)
                generation_output = model.generate(
                    input_ids=input,
                    generation_config=sampling,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=max_new_token,
                    pad_token_id=tokenizer.eos_token_id
                )
            generation_output_decoded = tokenizer.decode(generation_output.sequences[0])
            # print(generation_output_decoded)
            split = prompter.template["response_split"]
            ans = generation_output_decoded.split(split)[-1].strip()
            if len(ans) <=0 or len(tgt_ans) <=0:
                continue
            rouge_score = rouge.get_scores(ans, tgt_ans, avg=True)# 计算rouge分数
            bleu = sentence_bleu(ans.split(), tgt_ans.split())
            score_rouge.append(rouge_score)
            score_bleu.append(bleu)
            if verbose:
                print('-------------------')
                print(test_prompt)
                print(f"Target:[{tgt_ans}]\nModel :[{ans}]")
                print(f"Rouge: {rouge_score}")
                print(f"Bleu : {bleu}")
                break

    scores_accum = {
        'rouge-1': {'r': [], 'p': [], 'f': []},
        'rouge-2': {'r': [], 'p': [], 'f': []},
        'rouge-l': {'r': [], 'p': [], 'f': []},
    }
    for scores in score_rouge:
        for rouge_type, values in scores.items():
            for metric, value in values.items():
                scores_accum[rouge_type][metric].append(value)
    
    ave_rouge = {
        'rouge-1': {'r': sum(scores_accum['rouge-1']['r']) / len(scores_accum['rouge-1']['r']),
                    'p': sum(scores_accum['rouge-1']['p']) / len(scores_accum['rouge-1']['p']),
                    'f': sum(scores_accum['rouge-1']['f']) / len(scores_accum['rouge-1']['f'])},
        'rouge-2': {'r': sum(scores_accum['rouge-2']['r']) / len(scores_accum['rouge-2']['r']),
                    'p': sum(scores_accum['rouge-2']['p']) / len(scores_accum['rouge-2']['p']),
                    'f': sum(scores_accum['rouge-2']['f']) / len(scores_accum['rouge-2']['f'])},
        'rouge-l': {'r': sum(scores_accum['rouge-l']['r']) / len(scores_accum['rouge-l']['r']),
                    'p': sum(scores_accum['rouge-l']['p']) / len(scores_accum['rouge-l']['p']),
                    'f': sum(scores_accum['rouge-l']['f']) / len(scores_accum['rouge-l']['f'])},
    }
    ave_bleu = sum(score_bleu)/len(score_bleu)
    if verbose:
        print('========== Accuracy ==========')
        print(f"Average Rouge: {ave_rouge}")
        print(f"Average Bleu : {ave_bleu}")
    
    return ave_rouge, ave_bleu

#model = LlamaForCausalLM.from_pretrained(
#model = AutoModelForCausalLM.from_pretrained(
#tokenizer = LlamaTokenizer.from_pretrained('linhvu/decapoda-research-llama-7b-hf')
#tokenizer = AutoTokenizer.from_pretrained('gpt2')
#tokenizer.pad_token_id = tokenizer.eos_token_id
#print(tokenizer.pad_token_id, tokenizer.eos_token_id)
'''tokenizer.pad_token_id = (
    0
)
tokenizer.padding_side = "left"'''
# = Prompter("alpaca")

'''for id in range(1, 10):
    single_weights = torch.load('./lora-shepherd-7b-autolora-1-4/10/0/local_output_{}/'.format(id))
    set_peft_model_state_dict(model_c, single_weights, "default")
    for param_tensor in model_c.state_dict():
        model.state_dict[param_tensor] += model_c.state_dict[param_tensor]

for param_tensor in model.state_dict():
    model.state_dict[param_tensor] = model.state_dict[param_tensor]/10.0'''

'''with open(count_fine_path, "a") as file:
    file.write(str({"dataset_name": data_path.split('/')[-1], "accuracy": score})+'\n')'''