from datetime import datetime
from src.audio.mic_stream import MicrophoneStream
import numpy as np

class Listener:
    
  def listen(self):
      # Initialize and start the microphone stream
      mic_stream = MicrophoneStream()
      print("* recording")
      
      frames = []  # Store audio frames
      silence_threshold = 0.01  # Adjust this value based on your needs
      silence_count = 0
      max_silence_count = 1  # About 1 second of silence (adjust as needed)
              
      # Process the audio stream
      for audio_chunk in mic_stream.generator():
          # Store the audio chunk
          frames.append(audio_chunk.tobytes())
          # Clean frames to avoid memory overflow
          
          # Check for silence
          max_amplitude = np.max(np.abs(audio_chunk))
          if max_amplitude > 0.1 and not mic_stream.capture:
              print(f"Start capture")
              mic_stream.capture = True
              silence_count = 0
              frames = frames[-(len(frames) - 4):]
              print(f"Frames: {len(frames)}")

          if max_amplitude < silence_threshold and mic_stream.capture:
              silence_count += 1            
          
          # If we detect enough silence, process the recorded audio
          if silence_count >= max_silence_count and len(frames) > max_silence_count:
              mic_stream.capture = False
              print(f"Stop capture")
              
              # Transcribe the audio
              print("\nTranscribing...")
              transcription = mic_stream.transcribe_from_buffer(frames)
              print(f"Transcription: {transcription}")
              
              # Clear frames for next recording
              frames = []
              silence_count = 0