import streamlit as st
from typing import BinaryIO, Optional
from os import environ, path
from io import BytesIO

from openai import OpenAI
from pydub import AudioSegment, silence, utils
from openai.types.audio import Transcription
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tempfile import NamedTemporaryFile

API_KEY = environ.get("OPENAI_API_KEY", "no-key")

SPEECH_TO_TEXT_MODEL = "whisper-1"
AUDIO_FILE_SIZE_LIMIT = 25 * 1024 * 1024
SUMMARIZE_MODEL = "gpt-4-1106-preview"

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


def extract_audio(uploaded_file: UploadedFile) -> BytesIO:
    """Extracts audio from uploaded file"""
    video = AudioSegment.from_file(uploaded_file)
    audio = BytesIO()

    video.export(audio, format="mp3")

    return audio


def split_audio(audio_buffer: bytes, number_of_chunks: int) -> list[bytes]:
    """Split audio into chunks"""
    file_chunks: list[bytes] = []

    segment: AudioSegment = AudioSegment.from_mp3(BytesIO(audio_buffer))
    chunks: list[AudioSegment] = silence.split_on_silence(
        segment, min_silence_len=1000, silence_thresh=-16, keep_silence=200
    )

    print("Split into", len(chunks), "chunks")

    # too low number of chunks, split by time
    if len(chunks) < number_of_chunks:
        print("Too low number of chunks, splitting by time", len(chunks))
        chunk_length_ms = len(segment) // number_of_chunks
        chunks = utils.make_chunks(segment, chunk_length_ms)
        print("Split into", len(chunks), "chunks")

    # merge chunks, if there are too many of them
    if len(chunks) > number_of_chunks * 2:
        print("Too many chunks, merging", len(chunks))

        chunks_to_merge = len(chunks) // number_of_chunks + 1
        output_chunks: list[AudioSegment] = []

        for i in range(0, len(chunks), chunks_to_merge):
            merged_chunk: AudioSegment = AudioSegment.empty()

            for j in range(i, min(i + chunks_to_merge, len(chunks))):
                merged_chunk += chunks[j]

            output_chunks.append(merged_chunk)

        print("Merged into", len(output_chunks), "chunks")
        chunks = output_chunks

    ctr: int = 0
    for chunk in chunks:
        print("Exporting chunk", ctr)
        ctr += 1

        chunk_file: BytesIO = BytesIO()
        chunk.export(chunk_file, format="mp3")
        file_chunks.append(chunk_file.getvalue())

    return file_chunks


def speech_to_text(path_to_file: BytesIO) -> str:
    """Transcribe file"""
    file_bytes: bytes = path_to_file.getvalue()
    file_size: int = len(file_bytes)
    file_chunks: list[bytes]

    if file_size < AUDIO_FILE_SIZE_LIMIT:
        file_chunks = [file_bytes]
    else:
        file_chunks = split_audio(file_bytes, file_size // AUDIO_FILE_SIZE_LIMIT + 1)

    client: OpenAI = OpenAI(api_key=API_KEY)
    text: list[str] = []
    ctr: int = 0

    for chunk in file_chunks:
        print("Transcribing chunk", ctr)
        ctr += 1

        chunk_file: BytesIO = BytesIO(chunk)
        chunk_file.name = "chunk.mp3"

        transcript: Transcription = client.audio.transcriptions.create(
            model=SPEECH_TO_TEXT_MODEL, file=chunk_file, response_format="text"
        )

        print(transcript.text)
        text.append(transcript.text)

    return "\n".join(text)


def summarize(conversation: str, system_prompt: str, prompt: str) -> str:
    """Return conversation summary"""
    client = OpenAI(api_key=environ.get("OPENAI_API_KEY", API_KEY))
    response = client.chat.completions.create(
        # model="text-davinci-002",
        model="gpt-4",
        # prompt=f"{system_prompt}\n{prompt}\n",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        # prompt=f"Проанализируй данный разговор и выдели 5 главных тезисов.\n{conversation}\n",
        max_tokens=512,
        temperature=0.7,
        stop=None,
    )
    # generated_summary = response["choices"][0]["text"].strip()
    generated_summary = response.choices[0].message.content
    print(generated_summary)
    return generated_summary


st.header("Корус: Отчёт по встрече")

if st.button("Начать заново", type="primary"):
    st.session_state.step = 1
    del st.session_state["audio_path"]
    del st.session_state["conversation"]
    del st.session_state["system_prompt"]
    del st.session_state["prompt"]

uploaded_file: Optional[UploadedFile] = st.file_uploader(
    "Файл в формате .mp4/.mp3", type=["mp4", "mp3"]
)

if "audio_stream" not in st.session_state:
    st.session_state.step = 1
    st.session_state.audio_stream = None
    st.session_state.conversation = None
    st.session_state.system_prompt = None
    st.session_state.prompt = None

if uploaded_file is not None:
    if st.session_state.step == 1 and uploaded_file.name.endswith(".mp3"):
        st.session_state.audio_stream = uploaded_file
        st.session_state.step = 2
        st.success("Аудио извлечено")

    if st.session_state.step == 1 and uploaded_file.name.endswith(".mp4") and st.button("Извлечь аудио из видео"):
        st.subheader("Извлечение аудио", divider=True)
        audio_stream: BytesIO = extract_audio(uploaded_file)
        st.session_state.audio_stream = audio_stream

        st.session_state.step = 2
        st.success("Аудио извлечено")

        st.download_button("Скачать аудио", audio_stream, file_name="audio.mp3", mime="audio/mp3")

    if st.session_state.step == 2 and st.button("Извлечь текст из аудио"):
        st.subheader("Извлечение текста из аудио", divider=True)
        text: str = speech_to_text(st.session_state.audio_stream)

        conversation = st.text_area("Извлечённый текст", text)
        system_prompt = st.text_area("Системный промпт", DEFAULT_SYSTEM_PROMPT)
        prompt = st.text_area("Промпт", DEFAULT_PROMPT)

        st.session_state.conversation = conversation
        st.session_state.system_prompt = system_prompt
        st.session_state.prompt = prompt

        st.session_state.step = 3

    if st.session_state.step == 3 and st.button("Создать саммари"):
        st.subheader("Создание саммари", divider=True)
        summary: str = summarize(
            st.session_state.conversation,
            st.session_state.conversation.system_prompt,
            st.session_state.prompt,
        )

        st.text("Результат:")
        st.write(summary)

        st.session_state.step = 4
