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
    messages.append({"role": "assistant", "content": text})
    print("Assistant:", text)
    return text

def get_llm_response(messages, model=None, max_tokens=512, temperature=0.7):
    """
    Use Google Gemini (Generative Language) REST API to generate assistant replies.
    Reads GEMINI_API_KEY or GOOGLE_API_KEY from env and GEMINI_MODEL (optional).
    Does not use the OpenAI API.
    """
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set in environment.")

    # model should be in the form "models/xyz"
    model = model or os.environ.get("GEMINI_MODEL", "models/gemini-2.5-flash")

    # build a simple prompt by concatenating the message history
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"System: {content}\n")
        elif role == "assistant":
            parts.append(f"Assistant: {content}\n")
        else:
            parts.append(f"User: {content}\n")
    # ask the model to continue as the assistant
    parts.append("Assistant: ")
    prompt_text = "\n".join(parts)

    url = f"https://generativelanguage.googleapis.com/v1/{model}:generate"
    payload = {
        "prompt": {"text": prompt_text},
        "temperature": float(temperature),
        "maxOutputTokens": int(max_tokens),
    }

    try:
        resp = requests.post(url, params={"key": api_key}, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    # extract generated text from known response shapes
    text = ""
    # common v1 shape: {"candidates":[{"output":"..."}], ...}
    if not text:
        try:
            text = data.get("candidates", [{}])[0].get("output", "") or ""
        except Exception:
            pass
    # alternative shape: {"candidates":[{"content":[{"text":"..."}]}]}
    if not text:
        try:
            candidates = data.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", [])
                if content and isinstance(content, list):
                    # join text fragments
                    text = "".join([c.get("text", "") for c in content]).strip()
        except Exception:
            pass
    # final fallback: top-level output
    if not text:
        text = (data.get("output") or "").strip()

    if not text:
        raise RuntimeError(f"Gemini returned no text. Response: {data}")

    # Save assistant reply into conversation context
    messages.append({"role": "assistant", "content": text})
    print("Assistant:", text)
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
            assistant_text = chat_with_openai(messages)

            # speak the assistant reply (uses your existing speak function)
            try:
                speak(assistant_text)
            except Exception as e:
                print("TTS failed:", e)

    except KeyboardInterrupt:
        print("\nConversation terminated by user.")