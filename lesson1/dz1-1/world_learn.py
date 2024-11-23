import transformers
import torch
from transformers import pipeline

def classify_text(text, model_name="bert-base-uncased-finetuned-sst-2-english"):
    """
    Классифицирует текст с помощью модели BERT.
    
    Важно: Убедитесь, что модель предварительно загружена.
    """
    try:
        # Пробуем загрузить модель, если она не найдена, то загружаем
        try:
            classifier = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1)
        except OSError as e:
            print(f"Ошибка загрузки модели: {e}")
            try:
                classifier = pipeline("text-classification", model=model_name, device=0 if torch.cuda.is_available() else -1, trust_remote_code=True)
            except Exception as e:
                print(f"Ошибка загрузки модели: {e}")
                return None
        

        result = classifier(text)
        return result[0]
    except Exception as e:
        print(f"Ошибка при классификации текста: {e}")
        return None
    

if __name__ == "__main__":
    texts_to_classify = [
        "This movie is amazing!",
        "The plot was terrible and the acting was poor.",
        "The movie was okay, nothing special.",
        "I absolutely loved this movie.",
        "This film was a total disaster."
    ]


    try:
        for text in texts_to_classify:
            result = classify_text(text)
            if result:
                print(f"Текст: {text}")
                print(f"Метка: {result['label']}, Вероятность: {result['score']:.4f}")
            else:
                print(f"Ошибка при классификации текста: {text}")
    except Exception as e:
        print(f"ОБЩАЯ ОШИБКА: {e}")
