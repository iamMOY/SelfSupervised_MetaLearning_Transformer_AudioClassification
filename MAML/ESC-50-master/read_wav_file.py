
import os
import sys
import librosa
import numpy as np
from tqdm import tqdm
import wave
from scipy.io.wavfile import read


path = "1-100032-A-0.wav"
w= wave.open(path, 'r')
print("Frequency: ", w.getframerate())


data = read(path)
print("Data shape: ", data[1].shape, "Sample rate: ", data[0],"Duration", data[1].shape[0]/data[0], "seconds")

