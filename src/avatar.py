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
from PIL import Image, ImageTk

_avatar = None
_avatar_lock = threading.Lock()
_stop_event = threading.Event()
_keyboard_thread = None

class AvatarWindow:
    def __init__(self, title="Agent"):
        self.root = tk.Tk()
        self.root.title(title)
        self.canvas_w = 320
        self.canvas_h = 320
        self.canvas = tk.Canvas(self.root, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack()

        # Load face image (fallback to a simple oval if image missing)
        assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets"))
        face_path = os.environ.get("AVATAR_FACE_PATH", os.path.join(assets_dir, "bennett.png"))
        mouth_path = os.environ.get("AVATAR_MOUTH_PATH", os.path.join(assets_dir, "mouth.png"))

        try:
            face_img = Image.open(face_path).convert("RGBA")
            face_img = face_img.resize((self.canvas_w, self.canvas_h), Image.LANCZOS)
            self.face_tk = ImageTk.PhotoImage(face_img)
            self.canvas.create_image(self.canvas_w//2, self.canvas_h//2, image=self.face_tk)
        except Exception as e:
            # fallback drawing if image missing
            print("Avatar: failed to load face image:", e)
            self.canvas.create_oval(40, 40, 280, 280, fill="#ffdcb3", outline="#e6b89c")
            self.canvas.create_oval(100, 110, 130, 140, fill="black")
            self.canvas.create_oval(190, 110, 220, 140, fill="black")

        # Load mouth overlay (transparent PNG). We'll resize it dynamically.
        try:
            self.mouth_base = Image.open(mouth_path).convert("RGBA")
            # initial mouth size: width relative to canvas
            mw = int(self.canvas_w * 0.25)
            mh = max(8, int(self.mouth_base.height * mw / max(1, self.mouth_base.width)))
            self.mouth_base = self.mouth_base.resize((mw, mh), Image.LANCZOS)
            self.mouth_tk = ImageTk.PhotoImage(self.mouth_base)
            # position mouth near lower center; adjust y as needed for your face image
            self.mouth_item = self.canvas.create_image(self.canvas_w//2, int(self.canvas_h*0.65), image=self.mouth_tk)
        except Exception as e:
            print("Avatar: failed to load mouth image:", e)
            # fallback: create a colored oval mouth item we can scale
            self.mouth_item = self.canvas.create_oval(120, 200, 200, 220, fill="#8b0000", outline="")

        self.text_var = tk.StringVar()
        self.label = tk.Label(self.root, textvariable=self.text_var, wraplength=300, justify="center")
        self.label.pack(pady=6)

        self.amp_q = queue.Queue()
        self._poll()
        self.thread = None

    def run(self):
        """Start Tk mainloop — must be called from the main thread (blocks)."""
        try:
            self.root.mainloop()
        except Exception:
            raise

    def stop(self):
        """Stop GUI mainloop cleanly (call from main thread or signal handler)."""
        try:
            self.root.quit()
        except Exception:
            pass

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
        # amp expected in [0, ~1]; map to mouth height
        min_h = 8
        max_h = 60
        h = min_h + (max_h - min_h) * min(1.0, amp * 8.0)

        # If mouth_item is an image, resize the base PNG and update it
        if hasattr(self, "mouth_base"):
            try:
                # preserve width, change height
                mw = self.mouth_base.width
                mouth_resized = self.mouth_base.resize((mw, max(1, int(h))), Image.LANCZOS)
                self.mouth_tk = ImageTk.PhotoImage(mouth_resized)
                self.canvas.itemconfigure(self.mouth_item, image=self.mouth_tk)
            except Exception as e:
                # fallback to oval resize
                coords = self.canvas.coords(self.mouth_item)
                cx1 = self.canvas_w//2 - 40
                cx2 = self.canvas_w//2 + 40
                top = int(self.canvas_h*0.65 - h/2)
                bottom = int(self.canvas_h*0.65 + h/2)
                try:
                    self.canvas.coords(self.mouth_item, cx1, top, cx2, bottom)
                except Exception:
                    pass
        else:
            # mouth_item is an oval; update its coords
            cx1 = self.canvas_w//2 - 40
            cx2 = self.canvas_w//2 + 40
            top = int(self.canvas_h*0.65 - h/2)
            bottom = int(self.canvas_h*0.65 + h/2)
            self.canvas.coords(self.mouth_item, cx1, top, cx2, bottom)

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
    """Signal the keyboard listener to stop."""
    _stop_event.set()
    if _keyboard_thread:
        try:
            _keyboard_thread.join(timeout=0.1)
        except Exception:
            pass