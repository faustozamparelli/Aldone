import numpy as np
import sounddevice as sd
import whisper as ws
import librosa

duration = 10
samplerate = 16000
recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()
recording = np.squeeze(recording)
recording_resampled = librosa.resample(recording, orig_sr=samplerate, target_sr=16000)
model = ws.load_model("base")
mel = ws.log_mel_spectrogram(recording).to(model.device)
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")
options = ws.DecodingOptions()
result = ws.decode(model, mel, options)
print(result.text)
