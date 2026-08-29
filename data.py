
import torch
import pickle
import os
import pandas as pd
import numpy as np
from torch.utils.data import ConcatDataset, Dataset


def load_datasets(args):

    train_dataset = BSRDataset(args, mode="train")
    valid_dataset = BSRDataset(args, mode="valid")
    test_rec_dataset = BSRTestDataset(args, task="rec")
    test_src_dataset = BSRTestDataset(args, task="src")

    return train_dataset, valid_dataset, test_rec_dataset, test_src_dataset

class BSRDataset(Dataset):
    def __init__(self, args, mode="train"):
        super().__init__()

        self.mode = mode
        self.data_root = args.data_path

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_data('train')
        elif self.mode == 'valid':
            self.inter_data = self._process_data('valid')
        elif self.mode == 'test':
            self.inter_data = self._process_data('test')
        else:
            raise NotImplementedError    
        
    def get_new_tokens(self):
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens
       
    def _process_data(self, mode):
        rec_file = os.path.join(self.data_root, mode + '_rec.pkl')
        src_file = os.path.join(self.data_root, mode + '_src.pkl')
        rec_data = pickle.load(open(rec_file, 'rb'))
        src_data = pickle.load(open(src_file, 'rb'))
        rec_data = pd.DataFrame(rec_data).T
        src_data = pd.DataFrame(src_data).T

        self.new_tokens = set()

        # iterate over the data
        inter_data = []
        # 处理推荐任务数据
        for index, row in rec_data.iterrows():
            one_dict = dict()
            one_dict['target_item'] = '<item_'+str(row['target_item'])+'>'
            history = row['item_list']
            history = ['<item_'+str(i)+'>' for i in history]
            one_dict['input'] = get_prompt(" ".join(history), 'rec')
            one_dict['task_type'] = 0  # 0表示推荐任务
            inter_data.append(one_dict)
            self.new_tokens.update(history)
            self.new_tokens.add(one_dict['target_item'])
        
        # 处理搜索任务数据
        for index, row in src_data.iterrows():
            one_dict = dict()
            one_dict['target_item'] = '<item_'+str(row['target_item'])+'>'
            history = row['item_list']
            history = ['<item_'+str(i)+'>' for i in history]
            one_dict['input'] = get_prompt(" ".join(history), 'src', row['query'])
            one_dict['task_type'] = 1  # 1表示搜索任务
            inter_data.append(one_dict)
            self.new_tokens.update(history)
            self.new_tokens.add(one_dict['target_item'])
        return inter_data
    
    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["input"], labels=d["target_item"], task_type=d["task_type"])


class BSRTestDataset(Dataset):
    def __init__(self, args, task="rec"):
        super().__init__()

        self.task = task
        self.data_root = args.data_path

        # load data
        if self.task == 'rec':
            self.inter_data = self._process_data('rec')
        elif self.task == 'src':
            self.inter_data = self._process_data('src')
        else:
            raise NotImplementedError    
        
    def get_new_tokens(self):
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens
       
    def _process_data(self, mode):
        if mode == 'rec':
            file = os.path.join(self.data_root, 'test_rec.pkl')
        elif mode == 'src':
            file = os.path.join(self.data_root, 'test_src.pkl')
        data = pickle.load(open(file, 'rb'))
        data = pd.DataFrame(data).T

        self.new_tokens = set()

        # iterate over the data
        inter_data = []
        if self.task == 'rec':
            for index, row in data.iterrows():
                raw_targets = row['target_item'] if isinstance(row['target_item'], list) else [row['target_item']]
                target_items = ['<item_'+str(i)+'>' for i in raw_targets]
                if not target_items:
                    continue

                one_dict = dict()
                one_dict['target_item'] = target_items
                history = row['item_list']
                history = ['<item_'+str(i)+'>' for i in history]
                one_dict['input'] = get_prompt(" ".join(history), 'rec')
                one_dict['task_type'] = 0  # 0表示推荐任务
                inter_data.append(one_dict)
                self.new_tokens.update(history)
                one_dict['neg_item'] = ['<item_'+str(i)+'>' for i in row['neg_note_idx']]
                one_dict['neg_item'].extend(target_items)
                self.new_tokens.update(target_items)
                self.new_tokens.update(one_dict['neg_item'])
        elif self.task == 'src':
            for index, row in data.iterrows():
                raw_targets = row['target_item'] if isinstance(row['target_item'], list) else [row['target_item']]
                target_items = ['<item_'+str(i)+'>' for i in raw_targets]
                if not target_items:
                    continue

                one_dict = dict()
                one_dict['target_item'] = target_items
                history = row['item_list']
                history = ['<item_'+str(i)+'>' for i in history]
                one_dict['input'] = get_prompt(" ".join(history), 'src', row['query'])
                one_dict['task_type'] = 1  # 1表示搜索任务
                inter_data.append(one_dict)
                self.new_tokens.update(history)
                one_dict['neg_item'] = ['<item_'+str(i)+'>' for i in row['neg_note_idx']]
                one_dict['neg_item'].extend(target_items)
                self.new_tokens.update(target_items)
                self.new_tokens.update(one_dict['neg_item'])
        return inter_data
    
    def __len__(self):
        return len(self.inter_data)

    def __getitem__(self, index):
        d = self.inter_data[index]
        return dict(input_ids=d["input"], labels=d["target_item"], neg_item=d["neg_item"], task_type=d["task_type"])

def get_prompt(history, task, query=None):
    if task == 'rec':
        prompt = f"Below is the user's interaction history: {history}. Please recommend the next item the user is likely to click."
    elif task == 'src':
        prompt = f"Below is the user's interaction history: {history}. The user's search query is: {query}. Please predict the next item the user might click."
    else:
        raise ValueError(f"Invalid task: {task}")
    return prompt
