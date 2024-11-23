import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

data = {
    'text': [
        'Я люблю программирование.',
        'Погода сегодня прекрасная.',
        'Этот фильм был ужасен.',
        'Я обожаю читать книги.',
        'Спорт - это здорово!'
    ],
    'label': [1, 1, 0, 1, 1]  # 1 - положительный, 0 - отрицательный
}

df = pd.DataFrame(data)
df.to_csv('rus_text_classification.csv', index=False)

df = pd.read_csv('rus_text_classification.csv')
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

train_encodings = tokenizer.batch_encode_plus(train_texts, 
                                              add_special_tokens=True, 
                                              max_length=512, 
                                              padding='max_length', 
                                              truncation=True, 
                                              return_attention_mask=True, 
                                              return_tensors='pt')

val_encodings = tokenizer.batch_encode_plus(val_texts, 
                                             add_special_tokens=True, 
                                             max_length=512, 
                                             padding='max_length', 
                                             truncation=True, 
                                             return_attention_mask=True, 
                                             return_tensors='pt')
                                             
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels.values))
val_dataset = torch.utils.data.TensorDataset(val_encodings['input_ids'], val_encodings['attention_mask'], torch.tensor(val_labels.values))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
                                             
model = BertForSequenceClassification.from_pretrained('DeepPavlov/rubert-base-cased', num_labels=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Эпоха {epoch+1}, Потери: {total_loss / len(train_loader)}')

    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs.logits, 1)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / len(val_loader.dataset)
    print(f'Эпоха {epoch+1}, Точность на валидации: {accuracy:.4f}')


test_text = "Книги это зло"
test_encoding = tokenizer.encode_plus(test_text, 
                                       add_special_tokens=True, 
                                       max_length=512, 
                                       padding='max_length', 
                                       truncation=True, 
                                       return_attention_mask=True, 
                                       return_tensors='pt')

input_ids, attention_mask = test_encoding['input_ids'], test_encoding['attention_mask']

output = model(input_ids.to(device), attention_mask=attention_mask.to(device))
_, predicted = torch.max(output.logits, 1)
print(f'Предсказанная метка: {predicted.item()}')