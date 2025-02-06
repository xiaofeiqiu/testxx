import os
import threading
import time
import numpy as np
import sounddevice as sd
import whisper
import soundfile as sf
import tempfile

# Global buffer and lock to safely accumulate audio data
audio_buffer = []
buffer_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    """
    This callback is invoked by SoundDevice whenever new audio data is available.
    It appends a copy of the incoming data to a global buffer.
    """
    global audio_buffer
    # indata shape is (frames, channels); we assume a single channel.
    with buffer_lock:
        audio_buffer.append(indata.copy())

def transcribe_utterance(utterance_audio, fs, model):
    """
    Helper function that saves the provided audio to a temporary WAV file,
    runs Whisper transcription (with fp16 disabled), and prints the resulting text.
    This function is designed to run in its own thread.
    """
    # Save the utterance to a temporary WAV file.
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_filename = tmp.name
    sf.write(temp_filename, utterance_audio, fs, subtype='PCM_16')

    # Transcribe using Whisper (forcing CPU/FP32 mode).
    result = model.transcribe(temp_filename, fp16=False)
    os.remove(temp_filename)
    
    text = result.get("text", "").strip()
    if text:
        print("\nTranscription:", text)
    else:
        print("\n[No speech detected]")

def main(model_name="small", chunk_duration=2, fs=16000, silence_threshold=0.01, silence_duration=0.5):
    """
    Continuously records audio from the microphone. It checks the incoming audio
    buffer at regular intervals. When a period of silence (of length `silence_duration`)
    is detected at the end of the buffered audio and if speech is present in the buffer,
    the accumulated utterance is sent for transcription.
    
    Parameters:
      - model_name: The name of the Whisper model to use (e.g. "small").
      - chunk_duration: The approximate interval (in seconds) at which we check the buffer.
      - fs: Sampling rate in Hz (default 16000).
      - silence_threshold: Amplitude below which a segment is considered silent.
      - silence_duration: Duration (in seconds) of continuous silence needed to trigger transcription.
    """
    print(f"Loading Whisper model '{model_name}' on CPU (using FP32)...")
    model = whisper.load_model(model_name, device="cpu")
    
    print("Starting non-stop recording. Speak naturally and pause to trigger transcription.\n"
          "Press Ctrl+C to stop.\n")
    
    # Open the input stream in non-blocking mode using the callback.
    with sd.InputStream(channels=1, samplerate=fs, callback=audio_callback, dtype='float32'):
        try:
            while True:
                # Wait a short while before checking the audio buffer again.
                time.sleep(chunk_duration / 2)
                with buffer_lock:
                    if not audio_buffer:
                        continue  # No audio data yet.
                    
                    # Concatenate all accumulated audio chunks into one utterance.
                    utterance_audio = np.concatenate(audio_buffer)
                
                # Determine the number of samples corresponding to the silence check duration.
                num_silence_samples = int(silence_duration * fs)
                # Only check if we have enough samples.
                if len(utterance_audio) >= num_silence_samples:
                    # Extract the most recent samples.
                    recent_segment = utterance_audio[-num_silence_samples:]
                    # If the maximum amplitude in the recent segment is below the threshold,
                    # we consider it silent.
                    if np.max(np.abs(recent_segment)) < silence_threshold:
                        # Copy the current utterance for transcription and clear the buffer.
                        with buffer_lock:
                            audio_to_transcribe = np.concatenate(audio_buffer)
                            audio_buffer.clear()
                        
                        # Spawn a thread to transcribe the utterance.
                        threading.Thread(target=transcribe_utterance, args=(audio_to_transcribe, fs, model)).start()
        except KeyboardInterrupt:
            print("\nRecording stopped.")

if __name__ == "__main__":
    main(model_name="small", chunk_duration=2, fs=16000, silence_threshold=0.01, silence_duration=0.5)
