from torchtext.datasets import AG_NEWS
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from collections import Counter
from torchtext.data.utils import ngrams_iterator

def ensure_length(txt, out_len, pad_value):
    if len(txt) < out_len:
        txt = list(txt) + [pad_value] * (out_len - len(txt))
    else:
        txt = txt[:out_len]
    return txt

class LM_Dataset(Dataset):
    def __init__(self, tokens, labels, tokenizer, vocab_text, chunk_length=40, pad_value=16336):
        tokens_ids = [vocab_text(x) for x in tokens]
        tokens_ids = [ensure_length(x, chunk_length, pad_value) for x in tokens_ids]
        self.tokens = tokens_ids
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        return self.tokens[item], self.labels[item]

def make_data(text_with_labels, need_vocab=True, ngrams=1, minfreq=15):
    tokenizer = get_tokenizer("basic_english")
    tokens = []
    labels = []

    if need_vocab:
        counter = Counter()

    for label, line in text_with_labels:
        tokens.append(tokenizer(line))
        labels.append(label - 1)

        if need_vocab:
            if ngrams == 1:
                counter.update(tokenizer(line))
            elif ngrams > 1:
                counter.update(ngrams_iterator(tokenizer(line), ngrams=ngrams))

    if need_vocab:
        vocab_text = vocab(counter, min_freq=minfreq) # len(vocab) = 16336 if ngrams = 1
        pad_value = len(vocab_text)
        default_index = pad_value + 1
        # Добавляем индекс для неизвестных токенов
        vocab_text.set_default_index(default_index)
        # Добавляем токен для паддинга
        vocab_text.append_token("<pad>")

    if need_vocab:
        return tokens, labels, tokenizer, vocab_text, pad_value
    else:
        return tokens, labels

def encode(x, tokenizer, vocab_text):
    return vocab_text(tokenizer(x))

def decode(x, vocab_text):
    return [vocab_text.lookup_token(s) for s in x]

def load_data_news(ngrams=1, minfreq=15):
    train_iter = AG_NEWS(split='train')
    test_iter = AG_NEWS(split='test')

    news_text_train, news_labels_train, tokenizer, vocab_text, pad_value = make_data(train_iter, ngrams=ngrams, minfreq=minfreq)

    news_text_test, news_labels_test = make_data(test_iter, need_vocab=False)

    data_train = LM_Dataset(news_text_train, news_labels_train, tokenizer, vocab_text, pad_value=pad_value)
    data_test = LM_Dataset(news_text_test, news_labels_test, tokenizer, vocab_text, pad_value=pad_value)

    return data_train, data_test, tokenizer, vocab_text