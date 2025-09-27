import os
import sounddevice as sd
import torch
import torchaudio
import openai
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 🎤 Step 1: Record audio
def record_audio(filename="input.wav", duration=5, samplerate=16000):
    print("🎤 Recording... speak now!")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    waveform = torch.from_numpy(audio).squeeze()
    torchaudio.save(filename, waveform.unsqueeze(0), samplerate, format="wav")
    print("✅ Saved:", filename)
    return filename

# 📝 Step 2: Transcribe with Whisper
def transcribe(filename):
    with open(filename, "rb") as f:
        transcript = openai.audio.transcriptions.create(model="whisper-1", file=f)
    text = transcript.text
    print("🗣️ You said:", text)
    return text

# # 🤖 Step 3: Send to Gemini LLM
# def chat_with_gemini(prompt):
#     response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
#     text = response.text
#     print("🤖 Gemini:", text)
#     return text

# # 🔊 Step 4: Text-to-Speech
# def speak(text, filename="output.mp3"):
#     speech = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
#     with open(filename, "wb") as f:
#         f.write(speech.read())
#     print("🔊 Saved speech:", filename)
#     sd.play(torchaudio.load(filename)[0].T.numpy(), 24000)
#     sd.wait()
def test_google_api(test_prompt="Hello, test"):
    """Simple test of Google API with a prompt"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')
        response = model.generate_content(test_prompt)
        print(f"✅ Google API works! Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Google API failed: {e}")
        return False
    
# 🔁 Main loop
if __name__ == "__main__":
    audio_file = record_audio()
    test_google_api("How do you make a Chicago Deep Dish Pizza from scratch?")
    # user_text = transcribe(audio_file)
    # print(user_text)
    # llm_reply = chat_with_gemini(user_text)
    # speak(llm_reply)
