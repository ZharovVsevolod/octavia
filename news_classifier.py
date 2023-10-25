from octavia.utils.load_data import load_data_news
from octavia.utils.models import *
from octavia.utils.train import train, save_model
import wandb

# Загрузка данных
news_train, news_test, tokenizer, vocab_text = load_data_news()
len_vocab = len(vocab_text) + 1
print(f"Размер словаря = {len_vocab}")

# Создание модели
model = Classification_LSTM(hidden_size=150, embedding_dim=300, vocab_size=len_vocab, num_classes=4)
print(model)
print('Количество параметров', get_params_number(model))

# Подключаемся к wandb
wandb.login() # dec2ee769ce2e455dd463be9b11767cf8190d658
run = wandb.init(project="project_octavia", entity="wsewolod")

# Обучение модели
loss_history, best_model, optimizer = train(
    model=model,
    train_dataset=news_train,
    test_dataset=news_test,
    criterion=lm_cross_entropy,
    epoch_n=100,
    batch_size=32,
    device="cuda",
    lr_scheduler_default=lr_scheduler,
    need_wandb=False
)

# Закрываем пробег wandb
run.finish()

# Сохраняем лучшую модель
save_path = "octavia/model_weights"
save_model(save_path, best_model, optimizer, loss_history[-1], "last")