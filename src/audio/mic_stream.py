import pyaudio
import numpy as np
import whisper
import wave
from typing import Generator

class MicrophoneStream:
    def __init__(self, rate: int = 44100, chunk_size: int = 8192, channels: int = 2):
        self.rate = rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.p = None
        self.stream = None
        
        # Initialize Whisper model (using the smallest model for speed)
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Whisper model loaded!")
        
    def __enter__(self):
        self.p = pyaudio.PyAudio()
        info = self.p.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
            info = self.p.get_device_info_by_index(i)
            print(info)
        print(self.p.is_format_supported(self.rate, input_device=1, input_channels=self.channels, input_format=pyaudio.paFloat32))
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            input_device_index=1
        )
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
            
    def generator(self) -> Generator[np.ndarray, None, None]:
        """Generate chunks of audio data as numpy arrays."""
        while True:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.float32)
            yield audio_data
            
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

# Sample usage of the MicrophoneStream class
# def main():
#     # Initialize and start the microphone stream
#     with MicrophoneStream() as mic_stream:
#         print("* recording")
        
#         frames = []  # Store audio frames
#         silence_threshold = 0.01  # Adjust this value based on your needs
#         silence_count = 0
#         max_silence_count = 30  # About 1 second of silence (adjust as needed)
        
#         # Create a temporary directory for audio files
#         with tempfile.TemporaryDirectory() as temp_dir:
            
#             # Process the audio stream
#             for audio_chunk in mic_stream.generator():
#                 # Store the audio chunk
#                 frames.append(audio_chunk.tobytes())
                
#                 # Check for silence
#                 max_amplitude = np.max(np.abs(audio_chunk))
#                 if max_amplitude < silence_threshold:
#                     silence_count += 1
#                 else:
#                     silence_count = 0
                
#                 # If we detect enough silence, process the recorded audio
#                 if silence_count >= max_silence_count and len(frames) > max_silence_count:
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     temp_audio_file = os.path.join(temp_dir, f"audio_{timestamp}.wav")
                    
#                     # Save the audio chunk
#                     mic_stream.save_audio(frames, temp_audio_file)
                    
#                     # Transcribe the audio
#                     print("\nTranscribing...")
#                     transcription = mic_stream.transcribe_audio(temp_audio_file)
#                     print(f"Transcription: {transcription}")
                    
#                     # Clear frames for next recording
#                     frames = []
#                     silence_count = 0
                
#                 # Optional: Break after a certain duration or condition
#                 # if some_condition:
#                 #     break

# if __name__ == "__main__":
#     main()