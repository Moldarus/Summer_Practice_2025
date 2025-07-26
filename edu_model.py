import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import faiss
import os
from transformers import AutoTokenizer


class ModelTrainer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device=self.device
        )
        self.tokenizer = AutoTokenizer.from_pretrained("models/saiga_llama3_8b")

    def train(self, csv_path="training_data.csv", save_dir="knowledge_base_2"):
        """Создает векторное хранилище знаний"""
        os.makedirs(save_dir, exist_ok=True)

        # Загрузка данных
        df = pd.read_csv(csv_path, sep="|", quoting=3)
        texts = df['text'].tolist()

        # Векторизация
        embeddings = self.encoder.encode(texts, show_progress_bar=True)

        # Построение индекса FAISS
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings.astype('float32'))

        # Сохранение артефактов
        faiss.write_index(index, f"{save_dir}/index.faiss")
        df.to_pickle(f"{save_dir}/metadata.pkl")
        print(f"✅ Обучение завершено. Данные сохранены в {save_dir}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train()