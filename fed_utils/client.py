import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
from tqdm import tqdm
import torch
from DD import DDDataset
from DD import DDDataset, DD_DataCollatorForSeq2Seq
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

class GeneralClient:
    def __init__(self, client_id, model, data_path, output_dir, iid, usedata):
        self.client_id = client_id
        self.model = model
        if client_id == 0:
            self.local_data_path = os.path.join(data_path, "mix1.json") 
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        elif client_id == 1:
            self.local_data_path = os.path.join(data_path, "mix2.json")
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        elif client_id == 2:
            self.local_data_path = os.path.join(data_path, "mix3.json") 
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        elif client_id == 3:
            self.local_data_path = os.path.join(data_path, "mix4.json") 
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        elif client_id == 4:
            self.local_data_path = os.path.join(data_path, "mix5.json") 
            self.local_data = load_dataset("json", data_files=self.local_data_path)
        # self.local_data = self.local_data.rename_column('label', 'labels')
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))

    def preprare_local_dataset(self, generate_and_tokenize_prompt, local_val_set_size, usedata, tokenizer, useDD):  # 这里把它拆分成测试集和训练集然后变成token了，里面用max_len去cut了一下
        if useDD:
            if local_val_set_size > 0:
                local_train_val = self.local_data["train"].train_test_split(
                        test_size=local_val_set_size, shuffle=True, seed=42
                    )
            self.local_train_dataset = DDDataset(tokenizer, self.local_data_path, usedata)
            self.local_eval_dataset = None
        elif usedata == 'classification':
            def tokenize_function(examples):
                result = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
                result["labels"] = [examples['label']]
                return result
            if local_val_set_size > 0:
                local_train_val = self.local_data["train"].train_test_split(
                        test_size=local_val_set_size, shuffle=True, seed=42
                    )
                self.local_train_dataset = (
                    local_train_val["train"].shuffle().map(tokenize_function)
                )
                self.local_eval_dataset = (
                    local_train_val["test"].shuffle().map(tokenize_function)
                )
            else:
                self.local_train_dataset = self.local_data["train"].shuffle().map(tokenize_function)
            
                self.local_eval_dataset = None
            print(self.local_train_dataset)
            
        else:
            if local_val_set_size > 0:
                local_train_val = self.local_data["train"].train_test_split(
                    test_size=local_val_set_size, shuffle=True, seed=42
                )
                self.local_train_dataset = (
                    local_train_val["train"].shuffle().map(generate_and_tokenize_prompt)
                )
                self.local_eval_dataset = (
                    local_train_val["test"].shuffle().map(generate_and_tokenize_prompt)
                )
            else:
                self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt)
                
                self.local_eval_dataset = None
        self.local_val_set_size = local_val_set_size
        print(self.local_train_dataset)

    def build_local_trainer(self,
                            tokenizer,
                            local_micro_batch_size,
                            gradient_accumulation_steps,
                            local_num_epochs,
                            local_learning_rate,
                            group_by_length,
                            ddp,
                            usedata):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=0,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if self.local_val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=200 if self.local_val_set_size > 0 else None,
            save_steps=5000000,
            output_dir=self.local_output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if self.local_val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            dataloader_drop_last=False
        )
        if usedata == "classification":
            self.local_trainer = transformers.Trainer(model=self.model,
                                                    train_dataset=self.local_train_dataset,
                                                    eval_dataset=self.local_eval_dataset,
                                                    args=self.train_args,
                                                    data_collator=transformers.DataCollatorWithPadding(
                                                        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
                                                    )
        else:
            self.local_trainer = transformers.Trainer(model=self.model,
                                                  train_dataset=self.local_train_dataset,
                                                  eval_dataset=self.local_eval_dataset,
                                                  args=self.train_args,
                                                  data_collator=transformers.DataCollatorForSeq2Seq(
                                                      tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
                                                  )
                                                #   data_collator=DD_DataCollatorForSeq2Seq(
                                                #       tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
                                                #   )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(
            OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                        "default" in name))
        self.params_dict_new = OrderedDict((name, param.detach()) for name, param in self.model.named_parameters() if
                                           "default" in name)
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.params_dict_new, "default"
            )
        ).__get__(self.model, type(self.model))

    def train(self):
        self.local_trainer.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set, usedata, prompter, tokenizer):
        score1 = []
        for data_point in tqdm(self.local_eval_dataset):
            if usedata == "classification":
                if len(data_point["text"])==0:
                    continue

                test_prompt = prompter.generate_prompt(
                    data_point["instruction"],
                    data_point["text"],
                    '### Response:',
                )
            else:
                if len(data_point["input"])==0:
                    continue

                test_prompt = prompter.generate_prompt(
                    data_point["instruction"],
                    data_point["input"],
                    '### Response:',
                )

            with torch.no_grad():
                inputs = tokenizer(test_prompt, return_tensors="pt")
                input =inputs["input_ids"].to('cuda')
                    #print(tokenizer.eos_token_id, tokenizer.pad_token_id)
                generation_output = self.model(
                        input_ids=input
                    )
                # print(generation_output[0])
                # 将logits转换为类别标签
                predicted_label = torch.argmax(generation_output[0], dim=1).item()

                # 假设真实标签是以下列表
                true_label = data_point["label"]  # 你需要提供data_point["label"]的真实值
                # print(true_label)
                # print(predicted_label)
                # 比较预测的类别与真实标签
                is_correct = (predicted_label == true_label)
                # print(is_correct)
                score1.append(is_correct)
        s1 = sum(score1)/len(score1)
        print(f"Client {self.client_id} training accuracy is {s1}")
        
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = self.model.state_dict()
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        older_adapter_weight = get_peft_model_state_dict(self.model, self.params_dict_old, "default")
        set_peft_model_state_dict(self.model, older_adapter_weight, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})
        last_client_id = self.client_id

        return self.model, local_dataset_len_dict, previously_selected_clients_set, last_client_id, s1
