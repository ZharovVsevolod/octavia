from octavia import Light_Classification_LSTM
from octavia import NewsDataModule
import lightning as L

L.seed_everything(13, workers=True)

dm = NewsDataModule(32)
model = Light_Classification_LSTM(
    hidden_size=150,
    embedding_dim=300,
    vocab_size=dm.vocab_size,
    num_classes=dm.num_classes,
)
trainer = L.Trainer(
    max_epochs=2,
    accelerator="auto",
    devices=1,
    # callback = [],
)
trainer.fit(model, datamodule=dm)

# early_stopping
# lr callback
# In trainer model checkpoint callback

# L.module: on_test_epoch_start
# Wandb через lightning