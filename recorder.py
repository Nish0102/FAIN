import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os

OUTPUT_PATH = "data/output.wav"
DEFAULT_DURATION = 3
DEFAULT_FS = 16000
def record_audio(output_path=OUTPUT_PATH, duration=DEFAULT_DURATION, fs=DEFAULT_FS):
    devices = sd.query_devices()
    input_devices = [d for d in devices if d["max_input_channels"] > 0]
    if not input_devices:
        raise RuntimeError("No microphone found. Plug in a mic and try again.")

    print(f"🎙️ Recording for {duration}s at {fs}Hz...")

    try:
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="int16")
        sd.wait()
    except sd.PortAudioError as e:
        raise RuntimeError(f"Recording failed (PortAudio error): {e}")

    if np.max(np.abs(audio)) == 0:
        raise RuntimeError("Recording captured only silence. Check your microphone.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ✅ uses passed path
    write(output_path, fs, audio)

    size_kb = os.path.getsize(output_path) / 1024
    if size_kb < 1:
        raise RuntimeError(f"Saved file is too small ({size_kb:.1f} KB). Recording may be corrupt.")

    print(f"✅ Saved: {output_path} ({size_kb:.1f} KB)")
    return output_path  # ✅ returns the correct path