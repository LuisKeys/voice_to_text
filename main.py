from src.audio.mic_stream import MicrophoneStream
import os
from datetime import datetime

def main():
    with MicrophoneStream() as mic_stream:
        print("* recording")
        
        frames = []
        speech_frames_count = 0
        silence_frames_count = 0
        required_speech_frames = 20  # Minimum frames of speech to process
        max_silence_frames = 30  # Maximum consecutive silence frames
        temp_dir = './audio_data'
    
        for audio_chunk in mic_stream.generator():
            frames.append(audio_chunk.tobytes())
            
            # Check for speech
            if mic_stream.is_speech(audio_chunk):
                speech_frames_count += 1
                silence_frames_count = 0
            else:
                silence_frames_count += 1
            
            # Process audio when we have enough speech followed by silence
            if (speech_frames_count >= required_speech_frames and 
                silence_frames_count >= max_silence_frames):
                
                if len(frames) > max_silence_frames:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_audio_file = os.path.join(temp_dir, f"audio_{timestamp}.wav")
                    
                    mic_stream.save_audio(frames, temp_audio_file)
                    
                    print("\nTranscribing...")
                    transcription = mic_stream.transcribe_audio(temp_audio_file)
                    print(f"Transcription: {transcription}")
                    
                    frames = []
                    speech_frames_count = 0
                    silence_frames_count = 0

if __name__ == "__main__":
    main()