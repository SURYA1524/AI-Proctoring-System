import sounddevice as sd
import numpy as np

def detect_silence(duration=0.5, threshold=0.01):
    audio = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()
    volume = np.linalg.norm(audio) / len(audio)
    return volume < threshold