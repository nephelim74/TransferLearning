import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

# Данные (замените на ваши)
texts = [
    "I love this product!",
    "This movie was terrible.",
    "The food was amazing.",
    "I hate this game.",
    "This book is fantastic.",
    "I'm disappointed with this service."
]
labels = [1, 0, 1, 0, 1, 0]


# Разделение на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Токенизатор
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Модель
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# Обучение
class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Предобработка для обучения
train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors='pt')
train_dataset = SentimentDataset(train_encodings, train_labels)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, return_tensors='pt')
test_dataset = SentimentDataset(test_encodings, test_labels)


train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

optimizer = AdamW(model.parameters(), lr=5e-5)

# Обучение циклом
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Эпоха: {epoch}, Потери: {loss.item()}")


# Сохранение обученной модели
save_directory = "saved_model"
os.makedirs(save_directory, exist_ok=True)
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)


# Загрузка сохранённой модели и предсказание
loaded_model = BertForSequenceClassification.from_pretrained(save_directory)
loaded_tokenizer = BertTokenizer.from_pretrained(save_directory)

loaded_model.eval()
new_text = "I love this game"
new_encoding = loaded_tokenizer(new_text, return_tensors='pt', padding=True, truncation=True)
with torch.no_grad():
    outputs = loaded_model(**new_encoding)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()

if predicted_class == 1:
    print(f"'{new_text}' - Положительный оттенок.")
else:
    print(f"'{new_text}' - Отрицательный оттенок.")
