import os
import sounddevice as sd
import torch
import torchaudio
import openai
import google.generativeai as genai
import dotenv
dotenv.load_dotenv()
from openai import OpenAI
from prompts import AGENTPROMPT


genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Step 1: Record audio
def record_audio(filename="input.wav", duration=8, samplerate=16000):
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

# NVIDIA Audio2Face

# USER sound -> text, LLM call, Text -> Sound 

# # Step 3: text to text LLM
# def get_llm_response(prompt):
#     prompt = AGENTPROMPT + prompt + "Response: "
#     openai_key = os.environ.get("OPENAI_API_KEY")
#     if not openai_key:
#         raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or environment variables.")

#     client = OpenAI(api_key=openai_key)
#     resp = client.chat.completions.create(
#         model="gpt-3.5-turbo",  # change to the model you have access to
#         messages=[
#             {"role": "system", "content": "You are a patient, who has just walked into the Emergency Room."},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=512,
#         temperature=0.7,
#     )

#     # new client returns objects; extract the content
#     text = resp.choices[0].message.content.strip()
#     print("OpenAI:", text)
#     return text

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

# Step 4: Text-to-Speech
def speak(text, filename="output.mp3"):
    speech = openai.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
    with open(filename, "wb") as f:
        f.write(speech.read())
    print("🔊 Saved speech:", filename)
    sd.play(torchaudio.load(filename)[0].T.numpy(), 24000)
    sd.wait()

def run_conversation_loop():
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

    
# Main loop
if __name__ == "__main__":
    run_conversation_loop()
    # audio_file = record_audio()
    # audio_file = 'input.wav'
    # # test_google_api("How do you make a Chicago Deep Dish Pizza from scratch?")
    # user_text = transcribe(audio_file)
    # print(user_text)
    # llm_reply = get_llm_response(user_text)
    # speak(llm_reply)
