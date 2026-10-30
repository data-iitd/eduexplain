"""Persistent Indic Parler-TTS worker (run under venv-tts). JSON lines on stdin -> JSON lines on stdout.
Request: {"id","text","description","seed","out"}; reply: {"id","ok","sec","error"}. Logs to stderr only."""
import json
import sys
import time
import wave

import numpy as np


def log(*a):
    print("[parler_worker]", *a, file=sys.stderr, flush=True)


def main():
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    name = "ai4bharat/indic-parler-tts"
    import os
    dev = os.environ.get("PARLER_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    t0 = time.time()
    model = ParlerTTSForConditionalGeneration.from_pretrained(name, torch_dtype=dtype)
    try:
        model = model.to(dev).eval()
    except torch.OutOfMemoryError:
        log("CUDA OOM at load; falling back to CPU fp32")
        torch.cuda.empty_cache()
        dev, dtype = "cpu", torch.float32
        model = model.to(device="cpu", dtype=dtype).eval()
    tok = AutoTokenizer.from_pretrained(name)
    dtok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    sr = int(model.config.sampling_rate)
    log(f"loaded on {dev} {dtype} in {time.time() - t0:.1f}s")
    out = sys.stdout
    # (dev/model are rebound inside the loop on OOM fallback)
    sys.stdout = sys.stderr  # stray prints from libs must not corrupt the protocol
    print(json.dumps({"id": "ready", "ok": True, "sec": 0.0, "error": None, "sr": sr}), file=out, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        rid = None
        try:
            req = json.loads(line)
            rid = req.get("id")
            torch.manual_seed(int(req.get("seed", 0)))
            if dev.startswith("cuda"):
                torch.cuda.manual_seed_all(int(req.get("seed", 0)))
            def run():
                d = dtok(req["description"], return_tensors="pt").to(dev)
                p = tok(req["text"], return_tensors="pt").to(dev)
                with torch.inference_mode():
                    return model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                          prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask)
            try:
                gen = run()
            except torch.OutOfMemoryError:
                log("CUDA OOM during generate; moving model to CPU fp32")
                torch.cuda.empty_cache()
                dev, dtype = "cpu", torch.float32
                model.to(device="cpu", dtype=dtype)
                torch.manual_seed(int(req.get("seed", 0)))
                gen = run()
            a = gen.float().cpu().numpy().squeeze().reshape(-1)
            pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
            with wave.open(req["out"], "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(pcm.tobytes())
            reply = {"id": rid, "ok": True, "sec": len(pcm) / sr, "error": None}
        except Exception as e:  # keep worker alive
            log("error:", repr(e))
            reply = {"id": rid, "ok": False, "sec": 0.0, "error": repr(e)}
        print(json.dumps(reply), file=out, flush=True)


if __name__ == "__main__":
    main()
