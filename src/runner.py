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

    
# Main loop
if __name__ == "__main__":


    # ensure avatar exists (does not block)
    # avatar = get_avatar()

    # start keyboard listener in background
    # t1 = threading.Thread(target=start_keyboard_listener, daemon=True)
    # t1.start()

    # run the conversation loop in a background thread (non-daemon so we can cleanly join if needed)
    # t2 = threading.Thread(target=run_conversation_loop, daemon=False)
    # t2.start()
    run_conversation_loop()

    # # Run the GUI mainloop on the main thread (blocks here so process stays alive)
    # # avatar.run() must start Tk mainloop and block (defined in avatar.py)
    # avatar.run()


    
