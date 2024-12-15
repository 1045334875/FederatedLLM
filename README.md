# FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations
Code of paper : [FLoRA: Federated Fine-Tuning Large Language Models with Heterogeneous Low-Rank Adaptations](https://arxiv.org/pdf/2409.05976).

You can use this code to fine-tune LLMs with LoRA by WizardLLM dataset or other datasets.
The LoRA fine-tuning method includes FLoRA, FedIT, and Zero-Padding. You can also use heterogeneous LoRA rank settings in FLoRA and Zero-Padding.

## Requirments
Install all the packages from requirments.txt
* pip install -r requirements.txt
* git clone https://github.com/EleutherAI/lm-evaluation-harness
* cd lm-evaluation-harness
* pip install -e .

## Data
* The training dataset of WizardLLM has already been downloaded and split in ./data_wiz/ fold.
* If you want to use your dataset, use the same format as ./data_wiz/.

## Running the experiments
* To run the FLoRA algorithm (--stacking: True) and FedIT (--stacking False) in a homogeneous LoRA setting:
```
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FloRA-llama7b-wiz-homo/' --num_communication_rounds 3 --local_num_epochs 1 --stacking True
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FedIT-llama7b-wiz-homo/' --num_communication_rounds 3 --local_num_epochs 1 --stacking False
```
* To run the FLoRA algorithm (--stacking: True) and Zero-Padding (--stacking False --zero_padding True) in a heterogeneous LoRA setting:
```
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FloRA-llama7b-wiz-heter/' --num_communication_rounds 3 --local_num_epochs 1 --stacking True --heter True
python main.py --global_model 'huggyllama/llama-7b' --data_path  "./data_wiz" --output_dir './FedIT-llama7b-wiz-heter/' --num_communication_rounds 3 --local_num_epochs 1 --stacking False --heter True --zero_padding True
```

* To evaluate on LLM harness, try:
```
lm_eval --model_args pretrained=./FloRA-llama7b-wiz-homo/,parallelize=True,load_in_4bit=False, --tasks mmlu --num_fewshot 5 --batch_size 16 --output_path ../FloRA-llama7b-wiz-homo/
```
* To evaluate on MT-Bench, please follow the instructions on their websites: https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge
-----

# 新加
## 在服务器上跑

下面的命令就可以跑起来。

```
conda activate fedgpt
python main.py --global_model '/data/LLM_models/llama-7b' --data_path  "/data/ty/fedllm" --output_dir './FloRA-llama7b-test_eva/' 
python main.py --global_model '/data/LLM_models/llama-7b' --data_path  "/data/ty/fedllm" --output_dir './FloRA-llama7b-test_mini_eva/' 
python main.py --global_model '/data/LLM_models/llama-7b' --data_path  "/data/ty/fedllm/new" --output_dir './FloRA-llama7b-test/' 
python main.py --global_model "/data/LLM_models/opt-1.3b" --data_path  "/data/ty/fedllm/new" --output_dir './FloRA-opt_1.3b/' 

nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c3 --dataiid True --output_dir './FloRA-opt-iid_1/' > out-opt-iid1_1.txt  &
nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c3  --dataiid False --output_dir './FloRA-opt-non-iid_1/' > out-opt-non-iid1_1.txt  &

nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c5 --dataiid True --output_dir './FloRA-opt-iid_2/' > out-opt-iid2.txt  &

nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c5 --dataiid True --output_dir './FloRA-opt-iid_5/' > out-opt-iid5.txt  &
nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c3 --dataiid True --output_dir './FloRA-opt-iid_3/' > out-opt-iid3.txt  &

nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c5 --dataiid False --output_dir './FloRA-opt-non-iid_2/' > out-opt-non-iid2_1.txt  &

nohup python main.py --global_model "/data/LLM_models/opt-1.3b" --usedata c5 --dataiid False --output_dir './FloRA-opt-non-iid_5/' > out-opt-non-iid5.txt  &

nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3 --dataiid True --output_dir './FloRA-gemma-iid_3/' > out-gemma-iid3.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3 --dataiid False --output_dir './FloRA-gemma-non-iid_3/' > out-gemma-non-iid3.txt  &


nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3mini --dataiid True --output_dir './FloRA-gemma-iid_3mini/' > out-gemma-iid3mini_debug.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3mini --dataiid False --output_dir './FloRA-gemma-non-iid_3mini/' > out-gemma-non-iid3mini.txt  &


nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c5mini --dataiid True --output_dir './FloRA-gemma-iid_5mini/' > out-gemma-iid5mini.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c5mini --dataiid False --output_dir './FloRA-gemma-non-iid_5mini/' > out-gemma-non-iid5mini1.txt  &

nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata m3mini --dataiid True --output_dir './FloRA-gemma-iid_m3mini/' > out-gemma-iidm3mini.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata m3mini --dataiid False --output_dir './FloRA-gemma-non-iid_m3mini/' > out-gemma-non-iidm3mini.txt  &

11.16
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3_4k --dataiid True --local_num_epochs 1 --output_dir './FloRA-gemma-iid-c3_4k-lep1/' > out-gemma-iid-c3_4k-lep1.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3_4k --dataiid False --local_num_epochs 1  --output_dir './FloRA-gemma-non-iid-c3_4k-lep1/' > out-gemma-non-iid-c3_4k-lep1.txt  &


nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3_4k --dataiid True --local_num_epochs 3 --output_dir './FloRA-gemma-iid-c3_4k-lep3/' > out-gemma-iid-c3_4k-lep3.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c3_4k --dataiid False --local_num_epochs 3  --output_dir './FloRA-gemma-non-iid-c3_4k-lep3/' > out-gemma-non-iid-c3_4k-lep3.txt  &


nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c5_4k --dataiid True --local_num_epochs 1 --output_dir './FloRA-gemma-iid-c5_4k-lep1/' > out-gemma-iid-c5_4k-lep1.txt  &
nohup python main.py --global_model  "/data/ty/gemma-2b" --model_type gemma  --usedata c5_4k --dataiid False --local_num_epochs 1  --output_dir './FloRA-gemma-non-iid-c5_4k-lep1/' > out-gemma-non-iid-c5_4k-lep1.txt  &

nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama  --usedata m3mini --dataiid True --local_num_epochs 1 --output_dir './FloRA-llama-iid-m3mini-lep1/' > out-llama-iid-m3mini-lep1.txt  &
nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama  --usedata m3mini --dataiid False --local_num_epochs 1  --output_dir './FloRA-llama-non-iid-m3mini-lep1/' > out-llama-non-iid-m3mini-lep1.txt  &

nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama  --usedata m3mini --dataiid True --local_num_epochs 1 --output_dir './FloRA-llama-iid-m3mini-lep1_3/' > out-llama-iid-m3mini-lep1_3.txt  &
nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama  --usedata m3mini --dataiid False --local_num_epochs 1  --output_dir './FloRA-llama-non-iid-m3mini-lep1_3/' > out-llama-non-iid-m3mini-lep1_3.txt  &

===============================================分类任务=========================================================
nohup python main.py --global_model '/data/ty/gemma-2b' --model_type gemma --usedata classification --dataiid True --local_num_epochs 1 --output_dir './FloRA-gemma-iid-csf-lep1_3/' > out-gemma-iid-csf-lep1_3.txt  &
nohup python main.py --global_model '/data/ty/gemma-2b' --model_type gemma  --usedata classification --dataiid False --local_num_epochs 1  --output_dir './FloRA-gemma-non-iid-csf-lep1_3/' > out-gemma-non-iid-csf-lep1_3.txt  &


nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama   --usedata classification --dataiid True --local_num_epochs 1 --output_dir './FloRA-llama-iid-csf-lep1_3/' > out-llama-iid-m3mini-lep1_3.txt  &
nohup python main.py --global_model '/data/LLM_models/llama-7b' --model_type llama  --usedata m3mini --dataiid False --local_num_epochs 1  --output_dir './FloRA-llama-non-iid-m3mini-lep1_3/' > out-llama-non-iid-m3mini-lep1_3.txt  &
```

## 现有数据集

整理好的医学的，和之前的local training里面一样，在`/data/ty/fedllm`路径下，都是以下格式的:

```
    {
        "instruction": "Answer the following question based on the provided context.",
        "input": "How is a fecal occult blood test (stool test) used to diagnose gastritis?",
        "output": "This test checks for the presence of blood in your stool, a possible sign of gastritis."
    },
```

 MedQuAD我转为json的时候多加了一行class: health_care

本来转化后是这么大，我们把提取medical_train.json的前1/5，生成medical.json
- 16M     mashqa_train.json
- 28M     medical.json
- 25M     MedQuAD.json
- 588K    medical_test.json
- 133M    medical_train.json
- 596K    medical_valid.json

原始数据集
```
/data/ty/FALQU 法学领域的数据集10k
    ./Qrels/        tsv格式的问答匹配关系，四列分别为Question_id, 0, Answer_id, 1
    ./Topcis/       xml格式的问题，<ID>是Question_id, <BODY>是问题文本, <TITLE>是原文中的题目
    ./LawPosts.xml  xml格式的回答，<DOCNO>是answer_id, <DOC>是回答文本
    ./convert.py 我写的形成数据集的代码，目前在解读xml的时候会报错

/data/ty/MedQuAD 医疗领域 47k 这个整理好了，放在client_datasets下面了
/data/ty/mashqa_dataset  医疗领域35k，json格式
/data/ty/nature_question/ 综合问答


```

有一个医学的2k的LLeQA提交申请了还没通过
