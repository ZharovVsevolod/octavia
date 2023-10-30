from octavia.utils.load_data import NLP_DataModule
from octavia.utils.models import Light_Transformer_NLP, Transformer_Encoder
from torch import nn
import lightning as L

L.seed_everything(13, workers=True)

# dm = NLP_DataModule("./data", "DT14.txt", 128, 300, 1500)
# model = Light_Transformer_NLP(
#     vocab_size = dm.len_vocab,
#     embedding_size = 512,
#     backbone = Transformer_Encoder(
#         nn.TransformerEncoderLayer(
#             d_model = 512,
#             nhead = 16,
#             dim_feedforward = 1024,
#             dropout = 0.3
#         ),
#         num_layers=5
#     ),
#     emb_dropout = 0.2
# )

dm = NLP_DataModule("./data", "DT14.txt", 32, 100, 500)
model = Light_Transformer_NLP(
    vocab_size = dm.len_vocab,
    embedding_size = 128,
    backbone = Transformer_Encoder(
        nn.TransformerEncoderLayer(
            d_model = 128,
            nhead = 2,
            dim_feedforward = 256,
            dropout = 0.3
        ),
        num_layers=2
    ),
    emb_dropout = 0.2
)

trainer = L.Trainer(
    max_epochs=2,
    accelerator="auto",
    devices=1,
    # callback = [],
)
trainer.fit(model=model, datamodule=dm)

# early_stopping
# lr callback
# In trainer model checkpoint callback

# L.module: on_test_epoch_start
# Wandb через lightning