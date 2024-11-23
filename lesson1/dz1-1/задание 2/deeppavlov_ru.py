import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import os

# Важно: Подберите подходящую модель, поддерживающую русский язык!
MODEL_NAME = "DeepPavlov/rubert-base-cased"


def train_and_save_model(train_data, model_name=MODEL_NAME):
    texts = train_data['text']
    labels = train_data['label']

    tokenizer = BertTokenizer.from_pretrained(model_name)

    encoded_inputs = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')

    model = BertForSequenceClassification.from_pretrained(model_name)

    # Подготовка данных для обучения
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
    train_dataset = Dataset(train_encodings, train_labels)
    test_dataset = Dataset(test_encodings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    epochs = 3

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Эпоха: {epoch}, Потери: {loss.item():.4f}")


    # Сохранение обученной модели (вместо 'saved_model_ru')
    save_directory = "saved_model_ru"
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print("Модель и токенизатор сохранены.")

    return model, tokenizer


class Dataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].long() for key, val in self.encodings.items()} # важно! long()
        item['labels'] = torch.tensor(self.labels[idx]).long() # важно! long()
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


def predict_sentiment(model, tokenizer, text):
    try:
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        if 'input_ids' not in encoded_input:
            print("Ошибка: токенизатор не возвращает 'input_ids'")
            return None
        if 'attention_mask' not in encoded_input:
            encoded_input['attention_mask'] = torch.ones_like(encoded_input['input_ids'])

        with torch.no_grad():
            outputs = model(**encoded_input)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()
        return predicted_label
    except Exception as e:
        print(f"Ошибка при предсказании: {e}")
        return None



# Пример использования (замените на ваши данные)
# train_data = {
#     'text': [
#         'Я люблю программирование.',
#         'Погода сегодня прекрасная.',
#         'Этот фильм был ужасен.',
#         'Я обожаю читать книги.',
#         'Спорт - это здорово!',
#         'Фильм ужасен',
#         'Книга хорошая',
#         'Компьютер быстрый',
#         'Книги бывают ужасные',
#         'Я не люблю читать книги'
#     ],
#     'label': [1, 1, 0, 1, 1, 0, 1, 1, 0, 0]
# }

train_data = {
    'text': [
        'Я люблю программирование.',
        'Погода сегодня чудесная.',
        'Этот фильм ужасен.',
        'Обожаю читать книги.',
        'Спорт - это классно!',
        'Фильм отвратительный.',
        'Книга замечательная.',
        'Компьютер работает быстро',
        'Книги могут быть ужасными.',
        'Я ненавижу читать журналы.',
        'Не люблю читать, слишком скучно.',
        'Не нравится, когда книги слишком сложные.',
        'Я не люблю читать книги.',
        'Я люблю читать книги.',
        # ... еще больше примеро
    ],
    'label': [1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1]
}

# Обучение и сохранение модели
model, tokenizer = train_and_save_model(train_data)


# Предсказание для нового текста
test_text = "Я люблю читать книги."
predicted_label = predict_sentiment(model, tokenizer, test_text)

if predicted_label is not None:
    if predicted_label == 1:
        print(f"'{test_text}' - Положительный оттенок.")
    else:
        print(f"'{test_text}' - Отрицательный оттенок.")
