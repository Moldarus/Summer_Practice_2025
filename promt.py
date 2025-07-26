import re
import faiss
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer


class AugmentedGenerator:
    def __init__(self, model_path="models/saiga_llama3_8b", kb_path="knowledge_base_2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(
            'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            device=self.device
        )

        # Загрузка базы знаний
        self.index = faiss.read_index(f"{kb_path}/index.faiss")
        with open(f"{kb_path}/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)

        # Конфигурация модели
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.float16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, query, top_k=3):
        """Генерация ответа с улучшенным форматированием и источниками"""
        # Поиск релевантных фрагментов
        query_embed = self.encoder.encode([query])
        distances, indices = self.index.search(query_embed.astype('float32'), top_k)

        # Собираем контекст и источники
        context_parts = []
        sources_info = {}

        for i, idx in enumerate(indices[0]):
            source = self.metadata.iloc[idx]['source']
            text = self.metadata.iloc[idx]['text']
            sources_info[i + 1] = source  # Сохраняем соответствие номера материала и файла
            context_parts.append(f"[Материал {i + 1}]: {text}")

        context = "\n\n".join(context_parts)

        # Улучшенный промт с более естественным форматом ответа
        prompt = f"""Ты должен ответить на вопрос на основе предоставленных материалов. 
Ответ должен быть развернутым и содержательным, включать все релевантные данные из материалов.

Формат ответа:
1. Полный ответ на вопрос со всеми числовыми значениями и деталями
2. Для каждого факта укажи источник в формате: [Файл: "название_файла"]
3. При необходимости добавь краткое пояснение (1-2 предложения)

Если информации нет в материалах, ответь: "В предоставленных материалах нет информации по данному вопросу."

Материалы:
{context}

Вопрос: {query}

Развернутый ответ:"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        try:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,  # Увеличили лимит для более развернутых ответов
                temperature=0.4,  # Немного повысили температуру для более естественных ответов
                top_p=0.8,
                do_sample=True,
                repetition_penalty=1.2,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = full_response.split("Развернутый ответ:")[-1].strip()

            # Постобработка ответа:
            # 1. Заменяем ссылки на материалы на реальные названия файлов
            for mat_num, source in sources_info.items():
                answer = answer.replace(f"[Материал {mat_num}]", f'[Файл: "{source}"]')
                answer = answer.replace(f"[Источник: материал {mat_num}]", f'[Файл: "{source}"]')

            # 2. Удаляем лишние пробелы и переносы, но сохраняем структуру ответа
            answer = re.sub(r'\n\s+\n', '\n\n', answer)  # Удаляем лишние пустые строки
            answer = re.sub(r'[\t]+', ' ', answer)
            answer = re.sub(r'\s{2,}', ' ', answer).strip()

            # 3. Удаляем слишком частые повторения источников
            answer = re.sub(r'(\[Файл: "[^"]+"\]\s*){2,}', lambda m: m.group(1), answer)

            return answer

        except Exception as e:
            print(f"Ошибка генерации: {e}")
            return "Не удалось сформировать ответ."


if __name__ == "__main__":
    generator = AugmentedGenerator()
    print("Система готова к работе. Введите ваш вопрос или 'exit' для выхода.")

    while True:
        query = input("\nВаш вопрос: ").strip()
        if query.lower() == 'exit':
            break

        response = generator.generate(query)
        print("\nОтвет:", response if response else "Информация не найдена")