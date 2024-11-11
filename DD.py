import json
import math
from collections import defaultdict
from transformers import DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig

class DD(DataCollatorWithPadding):
    def __init__(self, tokenizer, pad_to_multiple_of=8, return_tensors="pt"):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        # 对features进行处理，例如填充、创建掩码等
        input_ids = [feature['input_ids'] for feature in features]
        attention_masks = [feature['attention_mask'] for feature in features]
        
        # 使用tokenizer的pad方法进行填充
        batch = self.tokenizer.pad(
            input_ids,
            attention_masks,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # 可以在这里添加更多的自定义逻辑
        
        return batch

    # # 使用自定义的data_collator
    # trainer = transformers.Trainer(
    #     model=self.model,
    #     train_dataset=self.local_train_dataset,
    #     eval_dataset=self.local_eval_dataset,
    #     args=self.train_args,
    #     data_collator=CustomDataCollator(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    # )

def binary_decomposition(n):
    """返回数字n的二进制分解"""
    decomposition = []
    while n > 0:
        decomposition.append(n & -n)  # 获取最低位的2的幂
        n -= decomposition[-1]
    return decomposition

def split_document(document, length):
    """根据文档长度的二进制分解拆分文档"""
    decomposition = binary_decomposition(length)
    sequences = []
    start = 0
    for power in reversed(decomposition):
        sequences.append(document[start:start + power])
        start += power
    return sequences

def distribute_to_buckets(sequences, buckets):
    """将序列根据长度分配到不同的桶中"""
    for seq in sequences:
        length = len(seq)
        # for i, bucket_power in enumerate(buckets):
        #     if length == 2 ** i:
        #         buckets[length].append(seq)
        #         break
    
        for i in range(len(buckets) - 1, -1, -1):
            bucket_power = 2 ** i
            if length == bucket_power:
                buckets[bucket_power].append(seq)
                break
    return buckets

def get_buckets(filename):
    # 假设我们的桶是2的幂次方，例如：2^0, 2^1, 2^2, ..., 2^9
    buckets = [[] for _ in range(12)]
    buckets = {2**i: bucket for i, bucket in enumerate(buckets)}
    

    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    tokenizer = LlamaTokenizer.from_pretrained('/data/LLM_models/llama-7b', token="your_token",)
    for item in data:
        document = item['input'] + item['output']
        result = tokenizer(
            document,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        length = len(result["input_ids"])
        sequences = split_document(result["input_ids"], length)
        buckets = distribute_to_buckets(sequences, buckets)
    compute_distribution(buckets, filename)


def compute_distribution(buckets, filename):
    # 从特定长度的桶中提取数据进行训练
    # 假设我们想要从长度为2^3的桶中提取数据进行训练
    # target_bucket_key = 2**2
    print(f"File  {filename} ")
    for i in range(10):
        # training_data = buckets[target_bucket_key]
        print(f"Length [2^{i}] = {len(buckets[2**i])}")

filename = ['/data/ty/fedllm/medical_train.json', '/data/ty/fedllm/mashqa_train.json',  '/data/ty/fedllm/MedQuAD_train.json']
for i in range(3):
    get_buckets(filename[i])
    
    