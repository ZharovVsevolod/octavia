from octavia.utils.load_data import NLP_DataModule
from octavia.utils.models import Light_Transformer_NLP, Transformer_Encoder, LightCheckPoint
from torch import nn
import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

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

wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
wandb_log = WandbLogger(project="project_octavia", name="octavia_small", save_dir="./model_weights")

checkpoint = LightCheckPoint(data_module=dm, phrase_for_gen="The Dark Tower was ", logger=wandb_log)
early_stopping = EarlyStopping(monitor="val_loss", patience=50)

trainer = L.Trainer(
    max_epochs=5,
    accelerator="auto",
    devices=1,
    default_root_dir="./model_weights",
    callbacks=[checkpoint, early_stopping],
    logger=wandb_log
    # fast_dev_run=5
)
trainer.fit(model=model, datamodule=dm)

wandb.finish()

# early_stopping
# lr callback
# In trainer model checkpoint callback

# L.module: on_test_epoch_start
# Wandb через lightning