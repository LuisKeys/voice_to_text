import pyaudio
import numpy as np
import whisper
import wave
from typing import Generator

class MicrophoneStream:
    def __init__(self):
        self.rate = 16000
        self.chunk_size = 8192
        self.channels = 1
        self.p = None
        self.stream = None
        
        # Initialize Whisper model (using the smallest model for speed)
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Whisper model loaded...")

        self.p = pyaudio.PyAudio()                
        device_index = self.find_device_index_by_name('Focusrite')

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=device_index
        )
        print("Stream opened...")
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
            
    def generator(self) -> Generator[np.ndarray, None, None]:
        while True:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            yield audio_data
            
    def save_audio(self, frames: list, filename: str):
        """Save audio frames to a WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            self.audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
            int16_audio_data = (self.audio_data * 32767).astype(np.int16)
            wf.writeframes(int16_audio_data)
            
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper."""
        result = self.model.transcribe(audio_file)
        return result["text"].strip()

    def find_device_index_by_name(self, name: str) -> int:        
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        device_index = 0
        for i in range(0, numdevices):
            info = self.p.get_device_info_by_index(i)
            if name.lower() in info.get('name', '').lower():
                device_index = i
                return device_index
            
        if device_index is None:
            print(f"No device with name containing '{name}' found")
            raise ValueError(f"No device with name containing '{name}' found")
            return None
