from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset

import lightning as L
from torch.utils.data import DataLoader, random_split

import youtokentome as yttm

import random
import numpy as np
import statistics
from sklearn.model_selection import train_test_split

def save_texts_to_file(texts, out_file):
    with open(out_file, 'w', encoding="utf-8") as outf:
        outf.write('\n'.join(texts))

class LanguageModelDataset(Dataset):
    """
    Dataset для языковой модели\n
    Возвращает предложение в двух экземплярах:
        - предложение без последнего токена (seed_part)
        - предложение без начального токена (target_part)
    Делит предложение на две части: "начало" и "конец"\n
    При созданиии LanguageModelDataset:
    :param token_ids: - токены предложений
    :param chunk_length: - длина предложения (для того чтобы обрезать или дополнить паддингом)
    :param pad_value: - значение паддинга, если предложение окажется короче, чем нужно\n
    При вызове LanguageModelDataset:
    :param index: - номер предложения
    :return: кортеж из двух элементов:
        - преложение токенов, что модель получит в качестве входа (seed_part)
        - предложение токенов, что модель будет пытаться получить (target_part)
    """
    def __init__(self, token_ids, chunk_length=100, pad_value=0):
        self.token_ids = token_ids
        self.chunk_length = chunk_length
        self.pad_value = pad_value

    def __len__(self):
        return len(self.token_ids)
    
    def ensure_length(self, txt, out_len, pad_value):
        if len(txt) < out_len:
            txt = list(txt) + [pad_value] * (out_len - len(txt))
        else:
            txt = txt[:out_len]
        return txt

    def __getitem__(self, item):
        text = self.token_ids[item]
        # Режем предложение в случайном месте по размеру чанка - обеспечим некоторую аугментацию данных
        start_i = random.randint(0, max(0, len(text) - self.chunk_length - 1))
        chunk = text[start_i : start_i + self.chunk_length + 1]
        
        # seed_part - то, что нейросеть будет видеть
        seed_part = chunk[:-1]
        # target_part - то, что нейросеть должна предсказать
        target_part = chunk[1:]

        # Дополняем паддингом предложение в случае, если оно короче, чем размер чанка
        seed_part = self.ensure_length(seed_part, self.chunk_length, self.pad_value)
        target_part = self.ensure_length(target_part, self.chunk_length, self.pad_value)

        seed_part = np.array(seed_part)
        target_part = np.array(target_part)

        return seed_part, target_part


class NLP_DataModule(L.LightningDataModule):
    def __init__(self, data_dir:str, data_name_file:str, batch_size:int,
                 chunks_size:int, len_vocab:int):
        super().__init__()
        self.data_dir = data_dir
        self.path_to_full_data = data_dir + "/" + data_name_file
        
        self.chunk_size_raw = chunks_size
        self.len_vocab = len_vocab
        self.batch_size = batch_size

    def load_chunks(self, path, more_space=True):
        # Открываем файл, читаем и записываем весь текст в массив по символам
        with open(path, 'r', encoding="utf-8") as fin:
            full_text = fin.read()
        
        # Делим массив по символам на предложения длиной в chunk_size_raw
        full_text = [full_text[start:start + self.chunk_size_raw] for start in range(0, len(full_text), self.chunk_size_raw // 2)]
        # Удаляем /n и заменяем их на пробел
        if more_space:
            full_text = [line.replace(u"\n", u" ") for line in full_text]
        
        return full_text
    
    def data_preprocessing(self):
        with open(self.path_to_full_data, encoding="utf-8") as input_file:
            text = input_file.read().split('\n')
            text = [line for line in text if line != ""]
            text = [line.replace(u'\xa0', u' ') for line in text]
        
        self.path_to_full_data = self.data_dir + "/data_edit.txt"
        save_texts_to_file(text, self.path_to_full_data)        
    
    def tokenization(self):
        self.token_file = self.data_dir + "/data_bpe.yttm"
        yttm.BPE.train(data=self.path_to_full_data, vocab_size=self.len_vocab, model=self.token_file)
        self.tokenizer = yttm.BPE(self.token_file)

    def prepare_data(self) -> None:
        # Преобработка сырых данных
        self.data_preprocessing()
        # Построение словаря токенизации
        self.tokenization()
        # Разбиение текста на тренировочную и тестовую часть
        full_text = self.load_chunks(self.path_to_full_data)
        text_train, text_test = random_split(full_text, [0.8, 0.2])
        self.path_to_train_data = self.data_dir + "/data_train.txt"
        save_texts_to_file(text_train, self.path_to_train_data)
        self.path_to_test_data = self.data_dir + "/data_test.txt"
        save_texts_to_file(text_test, self.path_to_test_data)
        # Нахождение оптимального размера чанка для токенизированного текста
        train_text = self.load_chunks(self.path_to_full_data)
        train_token_ids = self.tokenizer.encode(train_text, bos=True, eos=True)
        len_distribution = [len(sent) for sent in train_token_ids]
        # Находим моду распределения длин и округляем до большего десятка
        mode = statistics.mode(len_distribution)
        mode_round = np.round(mode, -1)
        self.chunk_size = mode_round if mode_round > mode else mode_round + 10
    
    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            text_train_full = self.load_chunks(self.path_to_train_data)
            text_train, text_val = train_test_split(text_train_full, test_size=0.3)

            train_token_ids = self.tokenizer.encode(text_train, bos=True, eos=True)
            self.train_dataset = LanguageModelDataset(train_token_ids, chunk_length=self.chunk_size)

            val_token_ids = self.tokenizer.encode(text_val, bos=True, eos=True)
            self.val_dataset = LanguageModelDataset(val_token_ids, chunk_length=self.chunk_size)
        
        if stage == "test" or stage is None:
            text_test = self.load_chunks(self.path_to_test_data)
            test_token_ids = self.tokenizer.encode(text_test, bos=True, eos=True)
            self.test_dataset = LanguageModelDataset(test_token_ids, chunk_length=self.chunk_size)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )