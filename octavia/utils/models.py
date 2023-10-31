import torch
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from torchmetrics.functional import accuracy
import numpy as np
from .generators import BeamGenerator

def get_params_number(model):
    return sum(t.numel() for t in model.parameters())

class Transformer_Encoder(nn.Module):
    """
    Класс трансформера-энкодера, нужен для транспонирования входных данных
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.net = nn.TransformerEncoder(*args, **kwargs)
        self.initialize_weights()
    
    def forward(self, src, *args, **kwargs):
        src = src.transpose(0, 1).contiguous()  # Shape = (MaxInLen, BatchSize, EmbSize)
        result = self.net(src, *args, **kwargs)  # Shape = (TargetLen, BatchSize, EmbSize)
        result = result.transpose(0, 1).contiguous()  # Shape = (BatchSize, TargetLen, EmbSize)
        return result
    
    def initialize_weights(self):
        for param in self.net.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

class Light_Transformer_NLP(L.LightningModule):
    """
    Основной класс для языковой модели\n
    Может принимать в себя любую форму реализации в качестве внутренней нейросети (backbone)\n
    :param vocab_size: - размер словаря токенов
    :param embedding_size: - размер эмбеддинга
    :param backbone: - нейронная модель для генерации текста
    :param emb_dropout: - размер "прореживания" (dropout) для эмбеддинга
    """
    def __init__(self, vocab_size:int, embedding_size:int, backbone:nn.Module, emb_dropout:int=0.0, 
                 lr:float=1e-5, l2_reg_alpha:float=0.0):
        super().__init__()
        self.save_hyperparameters()

        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.backbone = backbone
        self.out = nn.Linear(embedding_size, vocab_size)

        self.learning_rate = lr
        self.l2_reg_alpha = l2_reg_alpha
    
    def forward(self, seq):
        batch_size, max_in_length = seq.shape

        # Создание маски
        seed_padding_mask = seq == 0
        dependency_mask = self.mask_for_transformer(max_in_length).to(seq.device)
        
        # Эмбеддинг и позиционное кодирование
        seed_embs = self.embeddings(seq)  # Shape = (BatchSize, MaxInLen, EmbSize)
        pos_codes = self.positional_encoding(max_in_length)
        pos_codes = pos_codes.unsqueeze(0).to(seed_embs.device)
        seed_embs = seed_embs + pos_codes
        seed_embs = self.emb_dropout(seed_embs)

        # Shape =  (BatchSize, TargetLen, EmbSize)
        target_features = self.backbone(
            seed_embs,
            mask=dependency_mask,
            src_key_padding_mask=seed_padding_mask
        )
        logits = self.out(target_features)  # Shape =  (BatchSize, TargetLen, VocabSize)
        return logits
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
    
    def mask_for_transformer(self, length):
        # Создаём маску единиц
        full_mask = torch.ones(length, length)
        # Создаём диагональную маску с булевыми значениями
        ignore_mask = torch.tril(full_mask) < 1
        # Заполняем False диагональной маски в "маске единиц" значениями -inf
        full_mask.masked_fill_(ignore_mask, float('-inf'))
        # Остальное - нулями
        full_mask.masked_fill_(~ignore_mask, 0)
        return full_mask

    def positional_encoding(self, max_length):
        # Создаём массив, по которому будут генерироваться синусы и косинусы
        time = np.pi * torch.arange(0, max_length).float()
        freq_dividers = torch.arange(1, self.embedding_size // 2 + 1).float()
        inputs = time[:, None] / freq_dividers[None, :]
        
        # Берём значения синусов и косинусов в качестве ответа
        result = torch.zeros(max_length, self.embedding_size)
        result[:, 0::2] = torch.sin(inputs)
        result[:, 1::2] = torch.cos(inputs)
        return result

    def loss(self, pred, target):
        pred_flat = pred.view(-1, pred.shape[-1])  # BatchSize*TargetLen x VocabSize
        target_flat = target.view(-1)  # BatchSize*TargetLen
        return F.cross_entropy(pred_flat, target_flat, ignore_index=0)
    
    def lr_scheduler(self, opt):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5, verbose=True)
    
    def training_step(self, batch):
        x, y = batch

        x = torch.tensor(x).to(torch.long)
        y = torch.tensor(y).to(torch.long)

        pred = self(x)
        pred_loss = self.loss(pred, y)
        return pred_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        x = torch.tensor(x).to(torch.long)
        y = torch.tensor(y).to(torch.long)

        pred = self(x)
        pred_loss = self.loss(pred, y)
        self.log("val_loss", pred_loss, prog_bar=True)
        # preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, y, task="multiclass", num_classes=self.num_classes)
        # self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg_alpha)
        sch = self.lr_scheduler(optimizer)
        return (
            {'optimizer': optimizer, 'lr_scheduler': {"scheduler": sch, "monitor": "val_loss"}},
        )

class LightCheckPoint(L.Callback):
    def __init__(self, data_module, phrase_for_gen:str, beamsize=5, logger=None) -> None:
        super().__init__()
        self.phrase = phrase_for_gen
        self.data_module = data_module
        self.beamsize = beamsize
        if logger is not None:
            self.logger = logger
    
    def generation_and_log(self, pl_module: L.LightningModule):
        # Генерация текста для понимания, как модель тренируется
        beam_generator = BeamGenerator(pl_module, self.data_module.tokenizer)
        beam_gen_variants = beam_generator(
            seed_text=self.phrase, 
            beamsize=self.beamsize, 
            return_hypotheses_n=5,
            need_reweight=True
        )
        texts = []
        for score, pred_txt in beam_gen_variants:
            texts.append(pred_txt)
        self.logger.log_text(key="Generated phrase", columns=["text_1", "text_2", "text_3", "text_4", "text_5"], data=[texts])
        
    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.generation_and_log(pl_module)
    
    def on_test_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.generation_and_log(pl_module)