import streamlit as st
from typing import BinaryIO, Optional
from os import environ

from openai import OpenAI
from openai.types.audio import Transcription
from streamlit.runtime.uploaded_file_manager import UploadedFile

DEFAULT_SYSTEM_PROMPT = """Действуй как консультант, который анализирует содержимое разговора нескольких людей.
Проанализируй разговор между несколькими сотрудниками разных компаний.
Будь сосредоточенным и преданным своим целям. Твои последовательные усилия приведут к выдающимся достижениям. Тебе нужно быть уверенным в своих выводах."""

DEFAULT_PROMPT = """/////// Разговор ///////
НАЧАЛО РАЗГОВОРА

{conversation}

КОНЕЦ РАЗГОВОРА
---
Пожалуйста, проанализируй приведённый выше разговор и напиши список из 5-10 пунктов с главными выводами из этого разговора.
Это очень важно для моей карьеры.
Пожалуйста, действуй последовательно, будь строгим и честным. Ничего не придумывай и основывайся только на фактах. 
Проверяй свои выводы дважды. 
Твоя упорная работа приведёт к замечательным результатам. 
Я дам тебе чаевые в размере $1000 за самый лучший анализ!"""


# TODO: change me, use ffmpeg to extract audio from mp4
def extract_audio(uploaded_file: UploadedFile) -> str:
    """Extracts audio from uploaded file"""
    return "path/to/audio.wav"


# TODO: change me, see https://platform.openai.com/docs/guides/speech-to-text
def speech_to_text(path_to_file: str) -> str:
    """Transcribe file"""
    client: OpenAI = OpenAI(api_key=environ.get("OPENAI_API_KEY", "xxx"))

    audio_file: BinaryIO = open(path_to_file, "rb")
    transcript: Transcription = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )

    return transcript.text


# TODO: change me, see https://platform.openai.com/docs/guides/text-generation/chat-completions-api
def summarize(conversation: str, system_prompt: str, prompt: str) -> str:
    """Return conversation summary"""
    return ""

st.header("Корус: Отчёт по встрече")
st.button("Начать заново", type="primary")

uploaded_file: Optional[UploadedFile] = st.file_uploader("Файл в формате .mp4", type="mp4")

if uploaded_file is not None:
    st.subheader("Извлечение аудио", divider=True)

    audio_path: str = extract_audio(uploaded_file)
    st.write(audio_path)

    if st.button("Извлечь текст из аудио"):
        st.subheader("Извлечение текста из аудио", divider=True)
        text: str = speech_to_text(audio_path)

        conversation = st.text_area("Извлечённый текст", text)
        system_prompt = st.text_area("Системный промпт", DEFAULT_SYSTEM_PROMPT)
        prompt = st.text_area("Промпт", DEFAULT_PROMPT)

        if st.button("Создать саммари"):
            st.subheader("Создание саммари", divider=True)
            summary: str = summarize(conversation, system_prompt, prompt)

            st.text("Результат:")
            st.write(summary)
