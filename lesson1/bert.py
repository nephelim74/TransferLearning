import torch
from torch import nn
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW  # Импортируем AdamW из torch.optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

# Шаг 1: Загрузка предобученной модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Шаг 2: Создание пользовательского Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        return inputs, label

# Пример данных
texts = ["This is a positive example.", "This is a negative example."]
labels = [1, 0]  # Предположим, 1 - положительный, 0 - отрицательный

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# Создание DataLoader
train_dataset = TextDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Шаг 3: Определение модели
class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(bert_model.config.hidden_size, 2)  # 2 класса

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits

# Инициализация модели
model = BertClassifier()
optimizer = AdamW(model.parameters(), lr=1e-5)  # Используем AdamW из torch.optim

# Шаг 4: Обучение модели
model.train()
for epoch in range(3):  # Обучение на 3 эпохи
    for batch in train_loader:
        inputs, labels = batch
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)
        labels = labels.to(torch.long)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

# Шаг 5: Оценка модели
model.eval()
test_dataset = TextDataset(test_texts, test_labels)
test_loader = DataLoader(test_dataset, batch_size=2)

predictions, true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        input_ids = inputs['input_ids'].squeeze(1)
        attention_mask = inputs['attention_mask'].squeeze(1)

        outputs = model(input_ids, attention_mask)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.numpy())
        true_labels.extend(labels.numpy())

# Вывод точности
accuracy = accuracy_score(true_labels, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')