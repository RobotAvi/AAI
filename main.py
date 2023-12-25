#import streamlit as st
#from typing import BinaryIO, Optional
from os import environ
from openai import OpenAI
from pydub import AudioSegment
import speech_recognition as sr
from dotenv import load_dotenv

load_dotenv()
API_KEY = environ.get("OPENAI_API_KEY")

conversation = ""
DEFAULT_SYSTEM_PROMPT = """Действуй как консультант, который анализирует содержимое разговора нескольких людей.
Проанализируй разговор между несколькими сотрудниками разных компаний.
Будь сосредоточенным и преданным своим целям. Твои последовательные усилия приведут к выдающимся достижениям. Тебе нужно быть уверенным в своих выводах."""

def extract_audio(video_path: str, audio_output_path: str):
    video_audio = AudioSegment.from_file(video_path)
    video_audio.export(audio_output_path, format="wav")

def speech_to_text(path_to_file: str) -> str:
    client = OpenAI(api_key=environ.get("OPENAI_API_KEY", API_KEY))

    audio_file= open(path_to_file, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="text",

    )
    return(transcript)
def summarize(conversation: str, system_prompt: str, prompt: str) -> str:
    client = OpenAI(api_key=environ.get("OPENAI_API_KEY",API_KEY))
    response = client.chat.completions.create(
        #model="text-davinci-002",
        model="gpt-4",
        #prompt=f"{system_prompt}\n{prompt}\n",
         messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        #prompt=f"Проанализируй данный разговор и выдели 5 главных тезисов.\n{conversation}\n",
        max_tokens=512, 
        temperature=0.7, 
        stop=None 
    )
    #generated_summary = response["choices"][0]["text"].strip()
    generated_summary = response.choices[0].message.content
    print(generated_summary)
    return generated_summary

def main():
    video_path = './videos/video1.mp4'
    audio_output_path = './audio/audio.wav'
    
    # Extract audio from video
    extract_audio(video_path, audio_output_path)

    # Convert audio to text
    audio_text = speech_to_text(audio_output_path)

    conversation = audio_text
    print(conversation)
    DEFAULT_PROMPT = f"""
    НАЧАЛО РАЗГОВОРА

    {conversation}

    КОНЕЦ РАЗГОВОРА
    ----
    Пожалуйста, проанализируй приведённый выше разговор и напиши список из 5-10 пунктов с главными выводами из этого разговора.
    Это очень важно для моей карьеры.
    Пожалуйста, действуй последовательно, будь строгим и честным. Ничего не придумывай и основывайся только на фактах. 
    Проверяй свои выводы дважды. 
    Твоя упорная работа приведёт к замечательным результатам. 
    Я дам тебе чаевые в размере $1000 за самый лучший анализ!"""
    summarize(conversation, DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT)

if __name__ == "__main__":
    main()
