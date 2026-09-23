from pathlib import Path
import os, re, subprocess, sys, time
import requests
import torch
import torchaudio as ta
from pydub import AudioSegment, effects
from pydub.silence import detect_leading_silence
from chatterbox.tts import ChatterboxTTS

ROOT = Path(__file__).parent
OUT = ROOT / "output"
RAW = OUT / "raw"
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

SOCRATES_REF_URL = "https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/48402e0ec040427bb1848400600ea74e/id=4dc9fda8-4e3c-4d1f-8ddb-1fba283f4c5e.wav"
HUSSERL_REF_URL = "https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/0e2ff5b962084420879e076a2345d13f/id=4b038f34-545a-4bfd-b13b-4c3e7ee3ce09.wav"

def download(url, path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    path.write_bytes(r.content)

def prep_ref(url, name):
    src = OUT / f"{name}_source.wav"
    dst = OUT / f"{name}_ref.wav"
    download(url, src)
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-i",str(src),
        "-t","10","-ac","1","-ar","24000",
        "-af","highpass=f=60,lowpass=f=12000,loudnorm=I=-20:TP=-2:LRA=7",
        str(dst)
    ], check=True)
    return dst

def parse_dialogue(path):
    turns=[]
    for block in path.read_text(encoding="utf-8").split("\n\n"):
        block=block.strip()
        if not block:
            continue
        if block.startswith("SOCRATES:"):
            turns.append(("SOCRATES", block.split(":",1)[1].strip()))
        elif block.startswith("HUSSERL:"):
            turns.append(("HUSSERL", block.split(":",1)[1].strip()))
    return turns

def trim(seg):
    lead = detect_leading_silence(seg, silence_threshold=-43, chunk_size=10)
    rev = detect_leading_silence(seg.reverse(), silence_threshold=-43, chunk_size=10)
    start=max(0,lead-35)
    end=max(start+1,len(seg)-max(0,rev-55))
    return seg[start:end]

def pause_after(text):
    t=text.lower()
    ms=500
    if text.endswith("?"):
        ms=720
    if len(text) < 18:
        ms=650
    hard=[
        "what remains", "could it?", "there is the wound",
        "then here is the asymmetry", "one final question",
        "what comes after experience", "what will nothingness be like",
        "the disappearance of seeing", "now i cut you"
    ]
    if any(k in t for k in hard):
        ms=1100
    if "by succeeding" in t:
        ms=2400
    return ms

def clip_gain(seg, target=-20.0):
    if seg.dBFS == float("-inf"):
        return seg
    return seg.apply_gain(target - seg.dBFS)

print("Preparing reference voices...")
s_ref=prep_ref(SOCRATES_REF_URL, "socrates")
h_ref=prep_ref(HUSSERL_REF_URL, "husserl")

turns=parse_dialogue(ROOT/"dialogue.txt")
print("Turns:",len(turns))
assert len(turns) >= 70, f"Unexpectedly short dialogue: {len(turns)} turns"

print("Loading Chatterbox on CPU...")
torch.set_num_threads(max(1, os.cpu_count() or 4))
model=ChatterboxTTS.from_pretrained(device="cpu")
print("Model sample rate:",model.sr)

segments=[]
audition_segments=[]

for i,(speaker,text) in enumerate(turns,1):
    ref=s_ref if speaker=="SOCRATES" else h_ref
    # Socrates: more expressive and looser CFG for deliberate cadence.
    # Husserl: tighter, more controlled, slightly less expressive.
    exaggeration=0.57 if speaker=="SOCRATES" else 0.42
    cfg_weight=0.34 if speaker=="SOCRATES" else 0.46

    print(f"[{i:02d}/{len(turns)}] {speaker}: {text[:95]}", flush=True)
    with torch.inference_mode():
        wav=model.generate(
            text,
            audio_prompt_path=str(ref),
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    wav_path=RAW/f"{i:03d}_{speaker.lower()}.wav"
    ta.save(str(wav_path), wav.cpu(), model.sr)

    seg=AudioSegment.from_wav(wav_path)
    seg=trim(seg)
    seg=clip_gain(seg, -20.0 if speaker=="SOCRATES" else -19.5)
    # Very light dynamics only; intelligibility over "cinematic" processing.
    seg=effects.compress_dynamic_range(seg, threshold=-19.0, ratio=1.8, attack=7, release=85)

    segments.append(seg)
    p=AudioSegment.silent(duration=pause_after(text), frame_rate=seg.frame_rate)
    segments.append(p)
    if i <= 8:
        audition_segments += [seg,p]

def join(parts):
    out=AudioSegment.silent(duration=450, frame_rate=24000)
    for x in parts:
        out += x
    return out

aud=join(audition_segments)
master=join(segments)

aud_wav=OUT/"AUDITION_Socrates_Husserl.wav"
master_wav=OUT/"The_Last_Appearance_Chatterbox_Master.wav"
aud.export(aud_wav, format="wav")
master.export(master_wav, format="wav")

def final_master(inp, outp):
    subprocess.run([
        "ffmpeg","-y","-loglevel","error","-i",str(inp),
        "-af",
        "highpass=f=65,lowpass=f=14500,"
        "equalizer=f=3200:t=q:w=1.2:g=1.2,"
        "acompressor=threshold=-18dB:ratio=1.65:attack=8:release=100,"
        "loudnorm=I=-16:TP=-1.2:LRA=9",
        "-ar","44100","-ac","2","-codec:a","libmp3lame","-b:a","192k",
        str(outp)
    ], check=True)

aud_mp3=OUT/"AUDITION_Socrates_Husserl.mp3"
master_mp3=OUT/"The_Last_Appearance_Chatterbox_Master.mp3"
final_master(aud_wav,aud_mp3)
final_master(master_wav,master_mp3)

print("AUDITION:", aud_mp3, aud_mp3.stat().st_size)
print("MASTER:", master_mp3, master_mp3.stat().st_size)
print("MASTER_SECONDS:", len(master)/1000)
