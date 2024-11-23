import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Подготовка данных
texts = [
    "I love programming.",
    "Python is a great language.",
    "I hate bugs.",
    "Debugging is fun!",
    "I enjoy learning new things.",
    "I dislike error messages."
]
labels = [1, 1, 0, 1, 1, 0]  # 1 - положительный, 0 - отрицательный

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

# 2. Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# 3. Подготовка данных для DataLoader
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return {**inputs, 'labels': torch.tensor(label)}

train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

# 4. Обучение модели
optimizer = AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(3):  # Обучаем модель на 3 эпохи
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# 5. Оценка модели
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        outputs = model(**batch)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
        true_labels.extend(batch['labels'].tolist())

# 6. Вывод результатов
print(classification_report(true_labels, predictions))
