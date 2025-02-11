import pyaudio
import numpy as np
import whisper
import wave
from typing import Generator
from scipy.signal import butter, lfilter
import webrtcvad
import struct

class MicrophoneStream:
    def __init__(self, rate: int = 16000, chunk_size: int = 480, channels: int = 1):
        """Initialize the microphone stream.
        Note: Chunk size is set to 480 (30ms at 16kHz) for optimal VAD performance
        
        Args:
            rate: Sample rate in Hz (default: 16000)
            chunk_size: Number of frames per buffer (default: 480)
            channels: Number of channels (1=mono, 2=stereo) (default: 1)
        """
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.p = None
        self.stream = None
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Whisper model loaded!")
        
        # Initialize WebRTC VAD
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (most aggressive)
        
        # Initialize noise profile
        self.noise_profile = None
        self.noise_samples = []
        self.noise_sample_count = 50  # Number of frames to use for noise profiling
        
    def butter_bandpass(self, lowcut: float = 300.0, highcut: float = 3000.0, order: int = 5):
        """Design a butterworth bandpass filter."""
        nyq = 0.5 * self.rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to the data."""
        b, a = self.butter_bandpass()
        y = lfilter(b, a, data)
        return y
    
    def update_noise_profile(self, data: np.ndarray):
        """Update the noise profile using the current audio frame."""
        if len(self.noise_samples) < self.noise_sample_count:
            self.noise_samples.append(data)
        elif self.noise_profile is None:
            self.noise_profile = np.mean(np.stack(self.noise_samples), axis=0)
    
    def reduce_noise(self, data: np.ndarray) -> np.ndarray:
        """Reduce noise using spectral subtraction."""
        if self.noise_profile is not None:
            # Simple spectral subtraction
            clean_data = data - self.noise_profile
            # Apply a noise gate
            noise_gate = np.std(self.noise_profile) * 2
            clean_data[np.abs(clean_data) < noise_gate] = 0
            return clean_data
        return data

    def __enter__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
            
    def generator(self) -> Generator[np.ndarray, None, None]:
        """Generate chunks of audio data as numpy arrays with noise reduction."""
        print("Calibrating noise profile... Please stay quiet for a moment.")
        while True:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            
            # Update noise profile during initial calibration
            if len(self.noise_samples) < self.noise_sample_count:
                self.update_noise_profile(audio_data)
                print(f"Calibrating noise profile... {len(self.noise_samples)} / {self.noise_sample_count}", end='\r')
                continue
            
            # Apply noise reduction and filtering
            filtered_data = self.bandpass_filter(audio_data)
            clean_data = self.reduce_noise(filtered_data)
            
            # Normalize audio
            if np.max(np.abs(clean_data)) > 0:
                clean_data = clean_data / np.max(np.abs(clean_data))
            
            yield clean_data
        
            
    def is_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio frame contains speech using WebRTC VAD."""
        # Convert float32 -> int16
        audio_data_16 = (audio_data * 32768).astype(np.int16)
        raw_data = struct.pack("h" * len(audio_data_16), *audio_data_16)
        return self.vad.is_speech(raw_data, self.rate)
            
    def save_audio(self, frames: list, filename: str):
        """Save audio frames to a WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paFloat32))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
            
    def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio file using Whisper."""
        result = self.model.transcribe(audio_file)
        return result["text"].strip()
