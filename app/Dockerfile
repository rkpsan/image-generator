# Используйте официальный образ Python
FROM python:3.9-slim

# Установите рабочую директорию
WORKDIR /app

# Установите зависимости
COPY requirements.txt ./
RUN pip install fastapi uvicorn transformers accelerate diffusers peft && \
  pip install -r requirements.txt

# Копируйте исходный код приложения
COPY . .

# Укажите порт, на котором будет работать приложение
EXPOSE 8000

# Запустите приложение
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
