import os
import sounddevice as sd
import torch
import torchaudio
import openai
import google.generativeai as genai

# 🔑 API Keys
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"
openai.api_key = "YOUR_OPENAI_KEY"  # for Whisper STT + TTS

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎤 Step 1: Record audio
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎤 Recording... speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    waveform = torch.from_numpy(audio).squeeze()
    torchaudio.save(filename, waveform.unsqueeze(0), samplerate)
    print("✅ Saved:", filename)
    return filename

# 📝 Step 2: Transcribe with Whisper
def transcribe(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
    text = transcript.text
    print("🗣️ You said:", text)
    return text

# 🤖 Step 3: Send to Gemini LLM
def chat_with_gemini(prompt):
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    text = response.text
    print("🤖 Gemini:", text)
    return text

# 🔊 Step 4: Text-to-Speech
def speak(text, filename="output.mp3"):
    speech = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    with open(filename, "wb") as f:
        f.write(speech.read())
    print("🔊 Saved speech:", filename)
    sd.play(torchaudio.load(filename)[0].T.numpy(), 24000)
    sd.wait()

# 🔁 Main loop
if __name__ == "__main__":
    audio_file = record_audio()
    user_text = transcribe(audio_file)
    llm_reply = chat_with_gemini(user_text)
    speak(llm_reply)
