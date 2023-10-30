import wandb
from torch.utils.data import DataLoader
import torch
import copy
import tqdm
import traceback

def save_model(save_path:str, model, optimizer, score, name:str):
    torch.save({
        'model_state_dict': model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": score
    }, 
    save_path + f"model_{name}.tar")

def copy_data_to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return [copy_data_to_device(elem, device) for elem in data]
    raise ValueError('Недопустимый тип данных {}'.format(type(data)))


def train(model, train_dataset, test_dataset, criterion, optimizer_default=None, 
          epoch_n=25, need_wandb=True, lr=1e-5, batch_size=32, device=None,
          data_loader_default=DataLoader, shuffle_train=True, dataloader_workers_n=0,
          need_to_save_during_education=False, save_path=None, early_stopping_patience=20,
          lr_scheduler_default=None, early_optimizer=None):
    
    # Определяем device, cuda это или нет (если заранее не дано)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    model.to(device)

    # Кидаем в Adam параметры модели и lr (Adam, если не указано иного)
    if optimizer_default is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optimizer_default(model.parameters(), lr=lr)
    
    # Если мы продолжаем обучение, то тогда загружаем старый оптимизатор
    if early_optimizer is not None:
        optimizer.load_state_dict(early_optimizer)
    
    # Создаём lr_scheduler, если он есть
    if lr_scheduler_default is not None:
        lr_scheduler = lr_scheduler_default(optimizer)
    else:
        lr_scheduler = None
    
    # Кидаем данные в DataLoader с перемешиванием train
    train_dataloader = data_loader_default(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=dataloader_workers_n)
    test_dataloader = data_loader_default(test_dataset, batch_size=batch_size, shuffle=False, num_workers=dataloader_workers_n)

    # Определяем переменные для логирования и сохранения лучшей модели
    best_val_loss = float('inf')
    best_epoch_i = 0
    best_model = copy.deepcopy(model)
    loss_history = []

    # Начинаем обучение
    for epoch_i in range(1, epoch_n+1):
        try:
            print(f'Эпоха {epoch_i}')
            
            #-----Тренировка модели-----
            model.train()
            mean_train_loss = 0
            train_batches_n = 0
            for batch_i, (batch_x, batch_y) in enumerate(train_dataloader):                
                # Пакеты данных переносятся в нужном формате на нужный device
                # batch_x = torch.tensor(batch_x).to(torch.long)
                # batch_y = torch.tensor(batch_y).to(torch.long)
                batch_x = torch.stack(batch_x)

                batch_x = copy_data_to_device(batch_x, device)
                batch_y = copy_data_to_device(batch_y, device)

                # Делаем предсказание класса, считаем функцию потерь
                pred = model(batch_x)
                loss = criterion(pred, batch_y)

                # Градиентный спуск
                model.zero_grad()
                loss.backward()
                optimizer.step()

                # Служебное
                mean_train_loss += float(loss)
                train_batches_n += 1

            mean_train_loss /= train_batches_n
            print(f"Среднее значение функции потерь на обучении - {mean_train_loss}")
            if need_wandb:
                wandb.log({"Train loss" : mean_train_loss}, commit=False)

            #-----Валидация модели-----
            model.eval()
            mean_val_loss = 0
            val_batches_n = 0

            with torch.no_grad():
                for batch_i, (batch_x, batch_y) in enumerate(test_dataloader):                    
                    # Пакеты данных переносятся в нужном формате на нужный device
                    # batch_x = torch.tensor(batch_x).to(torch.long)
                    # batch_y = torch.tensor(batch_y).to(torch.long)
                    batch_x = torch.stack(batch_x)

                    batch_x = copy_data_to_device(batch_x, device)
                    batch_y = copy_data_to_device(batch_y, device)

                    # Делаем предсказание следующих токенов, считаем функцию потерь
                    pred = model(batch_x)
                    loss = criterion(pred, batch_y)

                    # Служебное
                    mean_val_loss += float(loss)
                    val_batches_n += 1

            mean_val_loss /= val_batches_n
            print(f"Среднее значение функции потерь на валидации - {mean_val_loss}")
            loss_history.append(mean_val_loss)
            if need_wandb:
                wandb.log({"Test loss" : mean_val_loss})

            # Если модель оказалась лучше, чем предыдущая лучшая
            if mean_val_loss < best_val_loss:
                # Сохраняем модель
                best_epoch_i = epoch_i
                best_val_loss = mean_val_loss
                best_model = copy.deepcopy(model)

                # Если need_to_save_during_education=True, то сохраняем эту модель ещё и на диск
                if need_to_save_during_education:
                    save_model(save_path, best_model, optimizer, best_val_loss, str(best_epoch_i))
                
                print("Новая лучшая модель!")
                if need_wandb:
                    wandb.log({"New best model" : 1})

            # Если модель долго не улучшается, значит, она достигла порога
            elif epoch_i - best_epoch_i > early_stopping_patience:
                print(f"Модель не улучшилась за последние {early_stopping_patience} эпох, прекращаем обучение")
                break
            
            elif need_wandb:
                wandb.log({"New best model" : 0})

            if lr_scheduler is not None:
                lr_scheduler.step(mean_val_loss)

            print()
        except KeyboardInterrupt:
            print("Досрочно остановлено пользователем")
            break
        except Exception as ex:
            print(f"Ошибка при обучении: {ex}\n{traceback.format_exc()}")
            break
    
    return loss_history, best_model, optimizer