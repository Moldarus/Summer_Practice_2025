import pandas as pd
from pathlib import Path
import re
from razdel import sentenize
import csv
import numpy as np

# =============================================
# 1. КОНСТАНТЫ И НАСТРОЙКИ
# =============================================
INPUT_CSV = "output_data.csv"  # Путь к исходному CSV
OUTPUT_CSV = "training_data.csv"
CHUNK_SIZE = 1000
MIN_CHUNK_LENGTH = 200
MAX_SENTENCES_PER_CHUNK = 5


# =============================================
# 2. ФУНКЦИИ ДЛЯ ОБРАБОТКИ ТЕКСТА
# =============================================
def clean_text(text):
    """Улучшенная очистка текста"""
    if pd.isna(text) or not text.strip():
        return ""

    text = str(text)
    text = re.sub(r'[^\w\s.,:;!?()-]', ' ', text, flags=re.UNICODE)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\d+\s*$', '', text)
    text = text.replace('«', '"').replace('»', '"').replace('—', '-').replace('–', '-')
    return text


def split_into_sentences(text):
    """Разбивка на предложения"""
    return [s.text for s in sentenize(text)] if text else []


def create_chunks(sentences):
    """Создание контекстных чанков с улучшенной логикой"""
    chunks = []
    current_chunk = []
    current_length = 0

    for sent in sentences:
        sent_length = len(sent)

        # Проверка условий для завершения чанка
        if (current_length + sent_length > CHUNK_SIZE or
                len(current_chunk) >= MAX_SENTENCES_PER_CHUNK):
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= MIN_CHUNK_LENGTH:
                chunks.append(chunk_text)
            current_chunk = []
            current_length = 0

        current_chunk.append(sent)
        current_length += sent_length

    # Обработка последнего чанка
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= MIN_CHUNK_LENGTH:
            chunks.append(chunk_text)

    return chunks


# =============================================
# 3. ОБРАБОТКА CSV ДАТАСЕТА
# =============================================
def process_csv():
    """Обработка CSV датасета"""
    data = []
    print(f"📖 Чтение датасета из {INPUT_CSV}...")

    try:
        # Чтение CSV с указанием заголовков
        df = pd.read_csv(
            INPUT_CSV,
            header=None,
            names=["PDF Name", "Extracted Text"],
            sep=",",  # Уточните разделитель при необходимости
            on_bad_lines='skip'
        )
    except Exception as e:
        print(f"❌ Ошибка чтения CSV: {str(e)}")
        return None

    print(f"🔍 Найдено {len(df)} документов для обработки")

    for index, row in df.iterrows():
        filename = row['PDF Name']
        text = row['Extracted Text']

        # Очистка текста
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue

        # Разбивка на предложения
        sentences = split_into_sentences(cleaned_text)
        if not sentences:
            print(f"⚠️ Не удалось извлечь предложения из: {filename}")
            continue

        # Создание чанков
        chunks = create_chunks(sentences)

        for i, chunk in enumerate(chunks):
            data.append({
                "source": filename,
                "text": chunk,
                "chunk_id": f"{Path(filename).stem}_{index}_{i}",
                "length": len(chunk),
                "num_sentences": chunk.count('.')  # Приблизительный подсчет
            })

        if index % 10 == 0:
            print(f"✅ Обработано документов: {index + 1}/{len(df)}")

    return data


# =============================================
# 4. СОХРАНЕНИЕ В CSV
# =============================================
def save_to_csv(data):
    """Сохранение данных с улучшенной обработкой ошибок"""
    if not data:
        print("⚠️ Нет данных для сохранения!")
        return False

    df = pd.DataFrame(data)

    print("\n📊 Статистика данных:")
    print(f"Всего чанков: {len(df)}")
    print(f"Уникальных документов: {df['source'].nunique()}")
    print(f"Средняя длина: {df['length'].mean():.1f} символов")
    print(f"Мин/Макс длина: {df['length'].min()}/{df['length'].max()}")

    try:
        df.to_csv(
            OUTPUT_CSV,
            index=False,
            encoding="utf-8-sig",
            sep="|",
            quoting=csv.QUOTE_NONNUMERIC
        )
        print(f"\n✅ Данные сохранены в {OUTPUT_CSV}")
        return True
    except Exception as e:
        print(f"❌ Ошибка при сохранении: {str(e)}")
        return False


# =============================================
# 5. ЗАПУСК
# =============================================
if __name__ == "__main__":
    print("🚀 Подготовка данных для обучения...")
    processed_data = process_csv()

    if processed_data:
        if save_to_csv(processed_data):
            print("\nПример обработанных данных:")
            sample = pd.DataFrame(processed_data).head(3)
            print(sample[['source', 'chunk_id', 'length']])