import requests
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torchvision.models import resnet18
import webbrowser
from pathlib import Path

# Загрузка модели ResNet-18
model = resnet18(pretrained=True)
model.eval()

# Преобразование изображения
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Загрузка словаря классов
def load_class_labels(file_path):
  """Загружает словарь классов из файла."""
  class_labels = {}
  with open(file_path, 'r') as f:
    for line in f:
      class_id, class_name = line.strip().split(':')
      class_labels[int(class_id)] = class_name
  return class_labels

# Функция для предсказания класса
def class_of_picture(image_path, class_labels):
  """Предсказывает класс изображения, выводит изображение и запускает поиск в браузере."""

  # Загрузка изображения
  if image_path.startswith("http"):
    response = requests.get(image_path, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw)
  else:
    image = Image.open(image_path)

  # Показ изображения
  image.show()

  # Предобработка изображения
  input_tensor = preprocess(image)

  # Добавление размерности для модели (batch size, channels, height, width)
  input_batch = input_tensor.unsqueeze(0)

  # Передача изображения в модель
  with torch.no_grad():
    output = model(input_batch)

  # Получение предсказания
  _, predicted = torch.max(output, 1)
  predicted_class = predicted.item()

  # Получение названия класса
  predicted_class_name = class_labels[predicted_class]

  # Вывод результата
  print(f"Предсказанный класс: {predicted_class_name}")

  # Поиск в браузере
  search_query = f"{predicted_class_name} images"
  webbrowser.open_new_tab(f"https://www.google.com/search?q={search_query}")

def load_image_path(file_path):
    with open(file_path, 'r') as file:
        return file.readline().strip()

# Пример использования функции
if __name__ == "__main__":
  script_directory = Path(__file__).resolve().parent
  print(f'{script_directory}\image_path_file')
  class_labels_file = f'{script_directory}\class_labels_file.txt' # путь к файлу с классами
  class_labels = load_class_labels(class_labels_file)
  # image_path = "https://www.kolomnadiesel.com/upload/iblock/7a7/7a73166f0423f1af560c4230ca7a3f5c.jpg" 
  image_path = load_image_path(f'{script_directory}\image_path_file')
  class_of_picture(image_path, class_labels)