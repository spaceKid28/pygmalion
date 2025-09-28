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

# # 📝 Step 2: Transcribe with Whisper
# def transcribe(filename):
#     openai_key = os.environ.get("OPENAI_API_KEY")
#     print("This is the Open AI Key: *************", openai_key)
#     if not openai_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or environment variables.")
#     client = OpenAI(api_key=openai_key)
#     with open(filename, "rb") as f:
#         transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
#     text = transcript.text
#     print("🗣️ You said:", text)
#     return text

# 📝 Step 2: Transcribe with Whisper (replaced to use Google Speech-to-Text via API key)
def transcribe(filename, samplerate=16000, language_code="en-US", timeout=60):
    """
    Transcribe a local WAV file using Google Speech-to-Text REST API with an API key.
    Requires GOOGLE_API_KEY in environment.
    """
    key = os.environ.get("GOOGLE_API_KEY")
    print("Using GOOGLE_API_KEY present:", bool(key))
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Add it to your .env or environment variables.")

    # read file and base64 encode
    with open(filename, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    url = f"https://speech.googleapis.com/v1/speech:recognize?key={key}"
    payload = {
        "config": {
            # let service auto-detect encoding; set sample rate to your recorder's rate
            "encoding": "ENCODING_UNSPECIFIED",
            "sampleRateHertz": samplerate,
            "languageCode": language_code,
            "enableAutomaticPunctuation": True,
        },
        "audio": {"content": audio_b64},
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Google STT request failed: {e}") from e

    data = resp.json()
    # assemble transcript text from results
    results = data.get("results", [])
    if not results:
        return ""

    parts = []
    for r in results:
        alt = r.get("alternatives", [])
        if alt:
            parts.append(alt[0].get("transcript", ""))

    text = " ".join([p for p in parts if p]).strip()
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
            audio_file = record_audio()           # record to input.wav (or configured filename)
            user_text = transcribe(audio_file)    # transcribe using your chosen method
            if not user_text:
                print("No transcription obtained; try again.")
                continue
            # delete the temp audio file now that transcription is done
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except OSError as e:
                print("Warning: failed to remove audio file:", e)

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