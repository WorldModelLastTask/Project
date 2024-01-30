import torch
import torch.nn as nn

from transformers import AutoModel

class LM(nn.Module):
    def __init__(self, state_dim=32, model_name="microsoft/deberta-v3-xsmall"):
        super(LM, self).__init__()
        self.lm = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(state_dim*384, 512)
    def forward(self, x):
        emb = self.lm(x).last_hidden_state
        emb = emb.reshape(emb.shape[0], -1)
        return self.fc(emb)