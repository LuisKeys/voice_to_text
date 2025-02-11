from src.audio.mic_stream import MicrophoneStream
import numpy as np
import os
from datetime import datetime

def main():
    # Initialize and start the microphone stream
    with MicrophoneStream() as mic_stream:
        print("* recording")
        
        frames = []  # Store audio frames
        silence_threshold = 0.01  # Adjust this value based on your needs
        silence_count = 0
        max_silence_count = 30  # About 1 second of silence (adjust as needed)
        
        # Create a temporary directory for audio files
        temp_dir = "audio_data"            
        # Process the audio stream
        for audio_chunk in mic_stream.generator():
            # Store the audio chunk
            frames.append(audio_chunk.tobytes())
            
            # Check for silence
            max_amplitude = np.max(np.abs(audio_chunk))
            if max_amplitude < silence_threshold:
                silence_count += 1
            else:
                silence_count = 0
            
            # If we detect enough silence, process the recorded audio
            if silence_count >= max_silence_count and len(frames) > max_silence_count:
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