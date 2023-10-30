import torch
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from torchmetrics.functional import accuracy

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

class Light_Classification_LSTM(L.LightningModule):
    def __init__(self, hidden_size, embedding_dim, vocab_size, num_classes, lr=1e-5):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size, num_layers=1)
        self.predictor = nn.Linear(hidden_size, num_classes)
        self.num_classes = num_classes
        self.learning_rate = lr
        self.val_loss = 2
    
    def forward(self, seq):
        emb = self.embeddings(seq)
        output, (hidden, _) = self.encoder(emb)
        answer = self.predictor(hidden.squeeze(0))
        return answer
    
    def loss(self, pred, target):
        return F.cross_entropy(pred, target)
    
    def lr_scheduler(self, opt):
        return torch.optim.lr_scheduler.CyclicLR(opt, self.learning_rate, self.learning_rate*100)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, verbose=True)
    
    def training_step(self, batch):
        x, y = batch

        x = torch.stack(x)

        logits = self(x)
        answer = self.loss(logits, y)
        return answer

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = torch.stack(x)

        logits = self(x)
        loss = lm_cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.val_loss = loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # sch = self.lr_scheduler(optimizer)
        return optimizer
        return (
            {'optimizer': optimizer, 'lr_scheduler': sch},
        )
        return (
            {'optimizer': optimizer, 'lr_scheduler': {"scheduler": sch, "monitor": self.val_loss}},
        )

def lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5, verbose=True)

def lm_cross_entropy(pred, target):
    return F.cross_entropy(pred, target)
    pred_flat = pred.view(-1, pred.shape[-1])  # BatchSize*TargetLen x VocabSize
    target_flat = target.view(-1)  # BatchSize*TargetLen
    return F.cross_entropy(pred_flat, target_flat, ignore_index=0)