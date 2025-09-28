import os
import sounddevice as sd
import torch
import torchaudio
from openai import OpenAI
# import google.generativeai as genai
import dotenv
import traceback
import threading
import requests
dotenv.load_dotenv()

from speechUtils import run_conversation_loop
from avatar import speak, start_keyboard_listener, get_avatar

from prompts import AGENTPROMPT


# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])



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
def get_gemini_models(api_key: str | None = None):
    """
    Return a list of (name, displayName) for models available to the given Gemini API key.
    Also prints the models (example usage moved inside this function).
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("No GEMINI_API_KEY found in environment and no api_key argument provided.")

    url = "https://generativelanguage.googleapis.com/v1/models"
    resp = requests.get(url, params={"key": api_key}, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    models = []
    for m in data.get("models", []):
        name = m.get("name") or m.get("model") or ""
        display = m.get("displayName") or m.get("description") or ""
        models.append((name, display))

    # moved example usage here
    try:
        for name, disp in models:
            print(f"{name} — {disp}")
    except Exception as e:
        print("Error:", e)

    return models



    
# Main loop
if __name__ == "__main__":
    get_gemini_models()

    

    # ensure avatar exists (does not block)
    # avatar = get_avatar()

    # start keyboard listener in background
    # t1 = threading.Thread(target=start_keyboard_listener, daemon=True)
    # t1.start()

    # run the conversation loop in a background thread (non-daemon so we can cleanly join if needed)
    t2 = threading.Thread(target=run_conversation_loop, daemon=False)
    t2.start()

    # # Run the GUI mainloop on the main thread (blocks here so process stays alive)
    # # avatar.run() must start Tk mainloop and block (defined in avatar.py)
    # avatar.run()


    
