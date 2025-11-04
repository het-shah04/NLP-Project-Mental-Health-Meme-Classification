import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch.nn as nn
from config import config

class RepresentationalModel(nn.Module):
    def __init__(self, config):
        super(RepresentationalModel, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(config['model'])

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        if 'pooler_output' in outputs:
            return outputs.pooler_output
        else:
            # fallback for models like RoBERTa that don't return pooler_output
            return outputs.last_hidden_state[:, 0]  # CLS token



