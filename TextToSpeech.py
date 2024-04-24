from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

# generate speech by cloning a voice using default settings
tts.tts_to_file(
    text="",
    file_path="/Users/faustozamparelli/Developer/AiDiary/lusClone.mp3",
    speaker_wav="/Users/faustozamparelli/Developer/AiDiary/lusAudio.wav",
    language="en",
)
