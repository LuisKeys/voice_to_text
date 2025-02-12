from datetime import datetime
import os
import numpy as np

from src.audio.mic_stream import MicrophoneStream

def main():
    # Initialize and start the microphone stream
    mic_stream = MicrophoneStream()
    print("* recording")
    
    frames = []  # Store audio frames
    silence_threshold = 0.01  # Adjust this value based on your needs
    silence_count = 0
    max_silence_count = 2  # About 1 second of silence (adjust as needed)
    
    # Create a temporary directory for audio files
    temp_dir = "./audio_data"
            
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

        if max_amplitude < silence_threshold and mic_stream.capture:
            silence_count += 1            
        
        # If we detect enough silence, process the recorded audio
        if silence_count >= max_silence_count and len(frames) > max_silence_count:
            mic_stream.capture = False
            print(f"Stop capture")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_audio_file = os.path.join(temp_dir, f"audio_{timestamp}.wav")
            
            # Save the audio chunk
            mic_stream.save_audio(frames, temp_audio_file)
            
            # Transcribe the audio
            print("\nTranscribing...")
            transcription = mic_stream.transcribe_audio(temp_audio_file)
            print(f"Transcription: {transcription}")
            
            # Clear frames for next recording
            frames = []
            silence_count = 0

if __name__ == "__main__":
    main()