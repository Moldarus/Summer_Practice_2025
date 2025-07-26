from transformers import AutoModelForCausalLM, AutoTokenizer

# Указываем модель
model_name = "IlyaGusev/saiga_llama3_8b"

# Загрузка модели и токенизатора
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Автоматическое распределение по доступным устройствам
    torch_dtype="auto"  # Автоматический выбор типа данных (float16/float32)
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Сохранение локально (опционально)
save_path = "models/saiga_llama3_8b"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ Модель и токенизатор успешно сохранены в {save_path}")