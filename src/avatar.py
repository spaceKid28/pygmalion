import io
import threading
import queue
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import tkinter as tk
import os
from openai import OpenAI

_avatar = None
_avatar_lock = threading.Lock()
_stop_event = threading.Event()
_keyboard_thread = None

class AvatarWindow:
    def __init__(self, title="Agent"):
        self.root = tk.Tk()
        self.root.title(title)
        self.canvas = tk.Canvas(self.root, width=320, height=320, bg="white")
        self.canvas.pack()
        self.canvas.create_oval(40, 40, 280, 280, fill="#ffdcb3", outline="#e6b89c")
        self.canvas.create_oval(100, 110, 130, 140, fill="black")
        self.canvas.create_oval(190, 110, 220, 140, fill="black")
        self.mouth = self.canvas.create_oval(120, 200, 200, 220, fill="#8b0000", outline="")
        self.text_var = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.text_var, wraplength=300, justify="center")
        self.label.pack(pady=6)
        self.amp_q = queue.Queue()
        self._poll()
        self.thread = threading.Thread(target=self.root.mainloop, daemon=True)
        self.thread.start()

    def _poll(self):
        try:
            amp = self.amp_q.get_nowait()
            if amp is None:
                self._set_mouth(0.0)
            else:
                self._set_mouth(amp)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _set_mouth(self, amp):
        min_h = 8
        max_h = 40
        h = min_h + (max_h - min_h) * min(1.0, amp * 8.0)
        cx1, cy1, cx2 = 120, 200, 200
        top = 200 - h / 2
        bottom = 200 + h / 2
        self.canvas.coords(self.mouth, cx1, top, cx2, bottom)

    def set_text(self, text):
        self.text_var.set(text)

    def get_queue(self):
        return self.amp_q

def get_avatar():
    global _avatar
    with _avatar_lock:
        if _avatar is None:
            _avatar = AvatarWindow(title="Agent")
        return _avatar

def close_avatar():
    global _avatar
    with _avatar_lock:
        if _avatar is not None:
            try:
                _avatar.root.destroy()
            except Exception:
                pass
            _avatar = None

def toggle_avatar():
    with _avatar_lock:
        if _avatar is None:
            get_avatar()
            return "ON"
        else:
            close_avatar()
            return "OFF"

def play_audio_bytes(audio_bytes: bytes, amp_q: queue.Queue):
    f = io.BytesIO(audio_bytes)
    data, sr = sf.read(f, dtype="float32")
    if data.ndim == 1:
        frames = data
    else:
        frames = np.mean(data, axis=1).astype("float32")
    chunk = int(sr * 0.05)
    def runner():
        try:
            sd.play(frames, samplerate=sr)
            for i in range(0, len(frames), chunk):
                frame = frames[i:i+chunk]
                amp = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
                try:
                    amp_q.put(amp, timeout=0.1)
                except Exception:
                    pass
                time.sleep(chunk / sr)
            sd.wait()
        finally:
            try:
                amp_q.put(None, timeout=0.1)
            except Exception:
                pass
    t = threading.Thread(target=runner, daemon=True)
    t.start()
    return t

def speak(text):
    """
    Synthesize speech with OpenAI TTS in-memory, play it, and animate avatar.
    Requires OPENAI_API_KEY in environment.
    """
    if not text:
        return
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env or environment variables.")
    client = OpenAI(api_key=openai_key)
    try:
        speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
        audio_bytes = speech.read()
    except Exception as e:
        print("TTS request failed:", e)
        return
    avatar = get_avatar()
    avatar.set_text(text)
    amp_q = avatar.get_queue()
    play_audio_bytes(audio_bytes, amp_q)

def _keyboard_listener():
    print("Keyboard commands: 'v' to toggle avatar, 'q' to quit.")
    while not _stop_event.is_set():
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if not cmd:
            continue
        if cmd in ("v", "avatar", "toggle"):
            state = toggle_avatar()
            print("Avatar:", state)
        if cmd in ("q", "quit", "exit"):
            _stop_event.set()
            break

def start_keyboard_listener():
    global _keyboard_thread
    if _keyboard_thread and _keyboard_thread.is_alive():
        return
    _keyboard_thread = threading.Thread(target=_keyboard_listener, daemon=True)
    _keyboard_thread.start()

def stop_keyboard_listener():
    _stop_event.set()
    if _keyboard_thread:
        _keyboard_thread.join(timeout=0.1)