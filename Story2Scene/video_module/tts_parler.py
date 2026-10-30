"""Indian-English female narrator via AI4Bharat Indic Parler-TTS (separate venv-tts worker process).
stdlib + numpy only; same interface as tts.KokoroTTS."""
import atexit
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_DESC = ("{voice} speaks in a clear, warm, moderately paced voice like a friendly teacher, "
                "with very clear audio and no background noise.")


class IndicParlerTTS:
    name = "indic-parler-tts"
    MIN_SPW, MAX_SPW, TRIES = 0.18, 0.75, 3

    def __init__(self, voice="Mary", speed=0.92, seed=0, python="venv-tts/bin/python",
                 worker="parler_worker.py", description=None):
        self.voice, self.speed, self.seed = voice, speed, seed
        self.python = str(python if os.path.isabs(str(python)) else HERE / python)
        self.worker = str(worker if os.path.isabs(str(worker)) else HERE / worker)
        self.description = description
        self._proc = None
        self._cache = {}
        self._n = 0
        self._tmp = tempfile.mkdtemp(prefix="parler_")
        atexit.register(self.close)

    # -- worker management
    def _start(self):
        if self._proc and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen([self.python, self.worker], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1, cwd=str(HERE))
        line = self._proc.stdout.readline()
        if not line or not json.loads(line).get("ok"):
            self.close()
            raise RuntimeError("parler worker failed to start (see stderr)")

    def close(self):
        p, self._proc = getattr(self, "_proc", None), None
        if p and p.poll() is None:
            try:
                p.stdin.close()
                p.wait(timeout=15)
            except Exception:
                p.kill()
                p.wait()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -- generation
    def _desc(self):
        return (self.description or DEFAULT_DESC).format(voice=self.voice)

    def _raw(self, text, seed):
        key = (text, self.voice, seed, self._desc())
        if key in self._cache:
            return self._cache[key]
        self._start()
        self._n += 1
        out = os.path.join(self._tmp, f"c{self._n}.wav")
        req = {"id": self._n, "text": text, "description": self._desc(), "seed": seed, "out": out}
        self._proc.stdin.write(json.dumps(req) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            self.close()
            raise RuntimeError("parler worker died")
        r = json.loads(line)
        if not r.get("ok"):
            raise RuntimeError(f"parler worker error: {r.get('error')}")
        with wave.open(out, "rb") as w:
            sr = w.getframerate()
            a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        os.remove(out)
        self._cache[key] = (a, sr)
        return a, sr

    def _best(self, text):
        nw = max(1, len(text.split()))
        best, best_dev = None, None
        for k in range(self.TRIES):
            a, sr = self._raw(text, self.seed + k)
            spw = len(a) / sr / nw
            if self.MIN_SPW <= spw <= self.MAX_SPW:
                return a, sr
            dev = abs(np.log(max(spw, 1e-3) / 0.4))  # distance from a typical 0.4 s/word
            if best is None or dev < best_dev:
                best, best_dev = (a, sr), dev
        return best

    def _atempo(self, a, sr, factor):
        p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-",
                            "-filter:a", f"atempo={factor:.5f}", "-f", "f32le", "-"],
                           input=a.astype("<f4").tobytes(), capture_output=True, check=True)
        return np.frombuffer(p.stdout, dtype="<f4").astype(np.float32)

    def synthesize(self, text, speed=None):
        a, sr = self._best(text)
        s = speed or self.speed
        if abs(s - 1.0) > 1e-3:
            a = self._atempo(a, sr, s)
        return np.asarray(a, dtype=np.float32).reshape(-1), int(sr)
