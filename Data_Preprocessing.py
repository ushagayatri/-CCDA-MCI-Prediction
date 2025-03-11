
import os
import librosa
import pandas as pd
import numpy as np
import speech_recognition as sr

def convert_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def preprocess_dataset(audio_folder):
    data = []
    for file in os.listdir(audio_folder):
        if file.endswith(".wav"):
            text = convert_audio_to_text(os.path.join(audio_folder, file))
            data.append({"filename": file, "transcription": text})
    return pd.DataFrame(data)

df = preprocess_dataset("data/raw/")
df.to_csv("data/processed/transcriptions.csv", index=False)
