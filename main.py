from vosk import Model, KaldiRecognizer
import wave

model = Model("model")
wf = wave.open("audio.wav", "rb")
rec = KaldiRecognizer(model, wf.getframerate())
print(rec.Result())