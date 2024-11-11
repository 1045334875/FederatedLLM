import json
import random
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from transformers import DataCollatorWithPadding
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, AutoConfig


class DDDataset(Dataset):
    def __init__(self, batch_token, tokenizer, filename, bucket_num = 12):
        # bucket_num  是默认的分桶数量
        # filename = ['/data/ty/fedllm/medical_train.json', '/data/ty/fedllm/mashqa_train.json',  '/data/ty/fedllm/MedQuAD_train.json']
        self.tokenizer = tokenizer
        self.batch_token = batch_token # 一个batch至少要采样多少token
        self.total_token_per_epoch, self.buckets = self.get_buckets(filename, bucket_num)# 数据集里的token总数, 分桶后的结果
        self.distribution = self.compute_distribution(self.buckets, bucket_num) # 统计不同

    def __getitem__(self, features):
        # 从self.buckets去token作为batch_token的样本
        # 桶的id根据curriculum随机选，选第几个batch也可以随机
        # 桶里的样本平均概率采样，桶的选择按桶样本量/总样本量的概率采样
        input_ids = [feature['input_ids'] for feature in features]
        attention_masks = [feature['attention_mask'] for feature in features]
        
        # 使用tokenizer的pad方法进行填充
        batch = self.tokenizer.pad(
            input_ids,
            attention_masks,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        choose_bucket = random.choices(range(self.bucket_num), weights=self.distribution, k=1)[0]
        target_bucket_key = 2**choose_bucket
        training_data = self.buckets[target_bucket_key] # 取出了那个桶里的所有token
        max_start_idx = len(training_data) - self.batch_token 
        start_idx = torch.randint(0, max_start_idx + 1, (1,)).item() # 随机选择一个起始索引
        
        # 根据起始索引取出一个连续的batch
        batch = self.training_data[start_idx:start_idx + self.batch_token]
        return batch

    def __len__(self):
        return self.total_token_per_epoch // self.batch_token
    

    def binary_decomposition(self, n):
        """返回数字n的二进制分解"""
        decomposition = []
        while n > 0:
            decomposition.append(n & -n)  # 获取最低位的2的幂
            n -= decomposition[-1]
        return decomposition

    def split_document(self, document, length):
        """根据文档长度的二进制分解拆分文档"""
        decomposition = self.binary_decomposition(length)
        sequences = []
        start = 0
        for power in reversed(decomposition):
            sequences.append(document[start:start + power])
            start += power
        return sequences

    def distribute_to_buckets(self, sequences, buckets):
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

    def get_buckets(self, filename, bucket_num):
        # 假设我们的桶是2的幂次方，例如：2^0, 2^1, 2^2, ..., 2^9
        buckets = [[] for _ in range(bucket_num)]
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
            sequences = self.split_document(result["input_ids"], length)
            buckets = self.distribute_to_buckets(sequences, buckets)
        
        return length, buckets


    def compute_distribution(self, buckets, bucket_num):
        # 从特定长度的桶中提取数据进行训练
        # 假设我们想要从长度为2^3的桶中提取数据进行训练
        len = []
        for i in range(bucket_num):
            len.append(len(buckets[2**i]))
        return len


    