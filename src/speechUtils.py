import os
import sounddevice as sd
import torch
import torchaudio
from openai import OpenAI
# import google.generativeai as genai
import dotenv
import traceback
import requests
import base64
import time
dotenv.load_dotenv()

from prompts import AGENTPROMPT

# Step 1: Record audio
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎤 Recording... speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    waveform = torch.from_numpy(audio).squeeze()
    torchaudio.save(filename, waveform.unsqueeze(0), samplerate, format="wav")
    print("✅ Saved:", filename)
    return filename

def speak_outloud(patient_text):
    """
    Convert patient text to speech using the best available TTS engine.
    
    Args:
        patient_text (str): The text for the AI patient to speak
    """
    print("🗣️ AI Patient says:", patient_text)
    
    # Method 1: Try OpenAI TTS (highest quality, requires API key)
    if _try_openai_tts(patient_text):
        return
    
    # Method 2: Try gTTS (good quality, requires internet)
    if _try_gtts(patient_text):
        return
    
    # Method 3: Fall back to pyttsx3 with optimized settings
    _fallback_pyttsx3(patient_text)

def _try_openai_tts(text):
    """Try OpenAI TTS - highest quality"""
    try:
        from openai import OpenAI
        import pygame
        
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not client.api_key:
            return False
            
        response = client.audio.speech.create(
            model="tts-1",  # or "tts-1-hd" for even higher quality
            voice="nova",   # Options: alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        # Save and play audio
        response.stream_to_file("temp_speech.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("temp_speech.mp3")
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
            
        # Clean up
        pygame.mixer.quit()
        os.remove("temp_speech.mp3")
        return True
        
    except Exception as e:
        print(f"OpenAI TTS failed: {e}")
        return False

def _try_gtts(text):
    """Try Google Text-to-Speech - good quality, free"""
    try:
        from gtts import gTTS
        import pygame
        
        tts = gTTS(text=text, lang='en', slow=False)
        tts.save("temp_gtts.mp3")
        
        pygame.mixer.init()
        pygame.mixer.music.load("temp_gtts.mp3")
        pygame.mixer.music.play()
        
        # Wait for playback to finish
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
            
        pygame.mixer.quit()
        os.remove("temp_gtts.mp3")
        return True
        
    except Exception as e:
        print(f"gTTS failed: {e}")
        return False

def _fallback_pyttsx3(text):
    """Fallback to pyttsx3 with optimized settings"""
    try:
        import pyttsx3
        
        engine = pyttsx3.init()
        
        # Get available voices
        voices = engine.getProperty('voices')
        if voices:
            # Try to find a better quality voice (look for SAPI5 on Windows)
            best_voice = None
            for voice in voices:
                if 'sapi5' in voice.id.lower() or 'microsoft' in voice.name.lower():
                    best_voice = voice
                    break
            
            if best_voice:
                engine.setProperty('voice', best_voice.id)
            else:
                engine.setProperty('voice', voices[0].id)
        
        # Optimize settings for better quality
        engine.setProperty('rate', 100)     # Slightly slower for clarity
        engine.setProperty('volume', 0.9)  # Not quite max to avoid distortion
        
        engine.say(text)
        engine.runAndWait()
        
    except Exception as e:
        print(f"⚠️ All TTS methods failed: {e}")

def transcribe(filename, samplerate=16000, language_code="en-US", timeout=60):
    """
    Transcribe a local WAV file using a local Whisper model (no neItwork API calls).
    - Requires the openai-whisper package: pip install -U openai-whisper
    - Set WHISPER_MODEL env var to choose model (e.g. "tiny", "base", "small", "medium", "large")
    - Uses GPU if available (torch.cuda.is_available()).
    """
    try:
        import whisper
    except Exception:
        raise RuntimeError(
            "Local transcription requires the openai-whisper package. "
            "Install with: pip install -U openai-whisper"
        )

    # choose model name from env or default
    model_name = os.environ.get("WHISPER_MODEL", "tiny")

    # choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Whisper model '{model_name}' on device: {device}")

    try:
        model = whisper.load_model(model_name, device=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{model_name}': {e}") from e

    # whisper will handle audio loading/resampling; pass language code as short form (e.g. 'en')
    lang = language_code.split("-")[0].lower() if language_code else None

    try:
        result = model.transcribe(filename, language=lang, task="transcribe")
    except Exception as e:
        raise RuntimeError(f"Local whisper transcription failed: {e}") from e

    text = (result.get("text") or "").strip()
    print("🗣️ You said:", text)
    return text

def chat_with_openai(messages, model=None, max_tokens=512, temperature=0.7):
    """
    Send the full messages list to OpenAI and append the assistant reply to messages.
    Returns the assistant text.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or environment variables.")
    client = OpenAI(api_key=openai_key)

    model = model or os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content.strip()

    # Save assistant reply into conversation context
    messages.append({"role": "patient", "content": text})
    print("AI Patient:", text)
    return text

# (removed local speak implementation; using avatar.speak from src/avatar.py)

def run_conversation_loop():
    from avatar import speak
    """
    Run the main loop: record -> transcribe -> append user message -> LLM -> speak.
    Conversation context is preserved in `messages`.
    Say 'quit' or 'exit' to stop the loop.
    """
    # start conversation with system prompt from AGENTPROMPT
    messages = [{"role": "system", "content": AGENTPROMPT}]

    try:
        while True:
            print("\n--- New turn ---")
            # audio_file = record_audio()           # record to input.wav (or configured filename)
            audio_file = "input.wav"
            user_text = transcribe(audio_file)    # transcribe using your chosen method
            if not user_text:
                print("No transcription obtained; try again.")
                continue
            # delete the temp audio file now that transcription is done
            # try:
            #     if os.path.exists(audio_file):
            #         os.remove(audio_file)
            # except OSError as e:
            #     print("Warning: failed to remove audio file:", e)

            # check for explicit exit commands from user
            if user_text.strip().lower() in ("quit", "exit", "stop"):
                print("Exiting conversation loop.")
                break

            # add user message to context and call the LLM
            messages.append({"role": "user", "content": user_text})
            ai_patient_reponse = chat_with_openai(messages)
            speak(ai_patient_reponse)
            
            # Wait for TTS to complete - estimate based on text length
            # Roughly 150 words per minute speaking rate
            word_count = len(ai_patient_reponse.split())
            estimated_duration = (word_count / 150) * 60  # Convert to seconds
            time.sleep(max(2, estimated_duration + 1))  # Minimum 2 seconds, plus 1 second buffer
            

            # # speak the assistant reply (uses your existing speak function)
            # try:
            #     speak(assistant_text)
            # except Exception as e:
                # print("TTS failed:", e)

    except KeyboardInterrupt:
        print("\nConversation terminated by user.")