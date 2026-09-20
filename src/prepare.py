import pandas as pd
import re

# 1. ЗАГРУЗКА ДАТАСЕТА
df = pd.read_csv('/kaggle/input/datasets/asan104/newsss/synthetic_text_data.csv')

print(f"Исходный размер: {df.shape}")

# 2. ВЫБОР ТЕКСТОВОЙ КОЛОНКИ
text_column = 'text'  # <-- замените на имя вашей колонки

# Убираем пропуски
df = df.dropna(subset=[text_column])

# 3. УДАЛЕНИЕ СПЕЦСИМВОЛОВ (включая точки и запятые)
def remove_special_chars(text):
    """
    Оставляет только буквы (русские и латинские), цифры и пробелы.
    Точки, запятые и все прочие знаки препинания удаляются.
    """
    # Разрешены: буквы, цифры, пробелы. Всё остальное (в т.ч. . и ,) — удаляется
    cleaned = re.sub(r'[^a-zA-Zа-яА-ЯёЁ0-9\s]', '', str(text))
    # Убираем множественные пробелы
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

df[text_column] = df[text_column].apply(remove_special_chars)

# 4. УДАЛЕНИЕ СТРОК КОРОЧЕ 50 СИМВОЛОВ
df = df[df[text_column].str.len() >= 50]

print(f"После удаления коротких строк: {df.shape}")

# 5. УДАЛЕНИЕ ДУБЛИКАТОВ
df = df.drop_duplicates(subset=[text_column])

print(f"После удаления дубликатов: {df.shape}")

# 6. СОХРАНЕНИЕ РЕЗУЛЬТАТА
df.to_csv('/kaggle/working/clean.csv', index=False)

print("Готово! Файл сохранён в /kaggle/working/cleaned_dataset.csv")