# AAI

## Зависимости

* Python 3.10
* [Poetry](https://python-poetry.org/)
* ffmpeg

## Установка

Выполнить следующие команды:

    poetry install --no-root

## Запуск

Выполнить следующие команды:

    poetry shell
    export OPENAI_API_KEY="тут указать ваш ключ OpenAI API"
    streamlit run main.py

## Что делать

- [ ] Реализовать метод `extract_audio` - извлекает аудио из mp4, нужно использовать библиотеку ffmpeg
- [ ] Реализовать метод `speech_to_text` - преобразовывает разговор в текст, нужно использовать OpenAI, модель Whisper
- [ ] Реализовать метод `summarize` - саммаризирует текст с помощью OpenAI и GPT-3.5/GPT-4
- [ ] (?) Реализовать `speech_to_text` с помощью локальной модели (Whisper)
- [ ] (?) Реализовать `summarize` с помощью локальной модели (Saiga 2 70B, Zephyr 7B или Mistral 7B)
- [ ] Поэкспериментировать с промптами
