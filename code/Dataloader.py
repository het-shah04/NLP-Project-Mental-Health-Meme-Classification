import torch
from transformers import AutoTokenizer

class AnxietyDataloader(torch.utils.data.Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.mappings = {
            "Nervousness": 0,
            "Lack of Worry Control": 1,
            "Excessive Worry": 2,
            "Difficulty Relaxing": 3,
            "Restlessness": 4,
            "Impending Doom": 5,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'ocr_text']
        label = self.data.loc[idx, 'meme_anxiety_categories']
        k_indices = self.data.loc[idx, 'indices']
        triples = self.data.loc[idx, 'triples']
        sample_id = self.data.loc[idx, 'sample_id']

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config['max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        encoded_triples = self.tokenizer.encode_plus(
            triples,
            add_special_tokens=True,
            max_length=self.config['max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "sample_id": sample_id,
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "label": torch.tensor(self.mappings[label], dtype=torch.long),
            "k_indices": k_indices,
            "triple_input_ids": encoded_triples['input_ids'].squeeze(0),
            "triple_attention_mask": encoded_triples['attention_mask'].squeeze(0),
        }

class DepressiveDataloader(torch.utils.data.Dataset):
    def __init__(self, config, data):
        self.config = config
        self.data = data
        self.tokenizer = AutoTokenizer.from_pretrained(config['model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.mappings = {
            "Concentration Problem": 0,
            "Lack of interest": 1,
            "Lack of Interest": 1,
            "Feeling down": 2,
            "Eating Disorder": 3,
            "Sleeping Disorder": 4,
            "Low Self-Esteem": 5,
            "Self Harm": 6,
            "Self-Harm": 6,
            "Feeling Down": 2,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.loc[idx, 'ocr_text']
        label = self.data.loc[idx, 'meme_depressive_categories']
        k_indices = self.data.loc[idx, 'indices']
        triples = self.data.loc[idx, 'triples']
        sample_id = self.data.loc[idx, 'sample_id']

        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.config['max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        encoded_triples = self.tokenizer.encode_plus(
            triples,
            add_special_tokens=True,
            max_length=self.config['max_len'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            "sample_id": sample_id,
            "input_ids": encoded['input_ids'].squeeze(0),
            "attention_mask": encoded['attention_mask'].squeeze(0),
            "label": torch.tensor(self.mappings[label], dtype=torch.long),
            "k_indices": k_indices,
            "triple_input_ids": encoded_triples['input_ids'].squeeze(0),
            "triple_attention_mask": encoded_triples['attention_mask'].squeeze(0),
        }
