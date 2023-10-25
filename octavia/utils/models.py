import torch
import torch.nn as nn
from torch.nn import functional as F

def get_params_number(model):
    return sum(t.numel() for t in model.parameters())

class Classification_LSTM(nn.Module):
    def __init__(self, hidden_size, embedding_dim, vocab_size, num_classes):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, num_classes)
    
    def forward(self, seq):
        emb = self.embeddings(seq)
        output, (hidden, _) = self.encoder(emb)
        answer = self.predictor(hidden.squeeze(0))
        return answer

def lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

def lm_cross_entropy(pred, target):
    return F.cross_entropy(pred, target)
    pred_flat = pred.view(-1, pred.shape[-1])  # BatchSize*TargetLen x VocabSize
    target_flat = target.view(-1)  # BatchSize*TargetLen
    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)