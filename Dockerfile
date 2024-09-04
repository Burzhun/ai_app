# Базовый образ Python с поддержкой Poetry
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y gcc libc-dev openjdk-11-jre-headless && apt-get clean;

# Установка инструмента poetry
RUN pip install poetry --no-cache-dir

# Копирование и установка зависимостей
WORKDIR /app
COPY poetry.lock pyproject.toml src/ /app/
RUN poetry config virtualenvs.create false \
    && poetry install --no-dev --no-interaction --no-ansi --no-root \
    && poetry run poe install_sage

    
# Переключение на финальный образ
FROM python:3.10-slim

# Копирование установленных зависимостей из предыдущего образа
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Копирование исходного кода приложения
COPY . /app
WORKDIR /app

# Указываем команду запуска приложения
CMD ["poetry run poe run_server"]
