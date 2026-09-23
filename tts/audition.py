from pathlib import Path
import requests, subprocess, torch, torchaudio as ta
from pydub import AudioSegment, effects
from chatterbox.tts import ChatterboxTTS

ROOT=Path(__file__).parent
OUT=ROOT/"output"
OUT.mkdir(parents=True,exist_ok=True)

SOCRATES_REF_URL="https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/48402e0ec040427bb1848400600ea74e/id=4dc9fda8-4e3c-4d1f-8ddb-1fba283f4c5e.wav"
HUSSERL_REF_URL="https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/0e2ff5b962084420879e076a2345d13f/id=4b038f34-545a-4bfd-b13b-4c3e7ee3ce09.wav"

TURNS=[
("SOCRATES","Then purge the picture. No darkness. No silence. No waiting. No empty duration. No sleeper. No observer whispering, I am gone. What remains?"),
("HUSSERL","No further experience. And no, that no-further-experience is never given, retained, remembered, fulfilled, or encountered by a later phase.")
]

def dl(url,path):
    r=requests.get(url,timeout=120); r.raise_for_status(); path.write_bytes(r.content)
def prep(url,name):
    src=OUT/f"{name}_src.wav"; dst=OUT/f"{name}_ref.wav"
    dl(url,src)
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(src),"-t","10","-ac","1","-ar","24000",
                    "-af","highpass=f=60,lowpass=f=12000,loudnorm=I=-20:TP=-2:LRA=7",str(dst)],check=True)
    return dst

sref=prep(SOCRATES_REF_URL,"socrates")
href=prep(HUSSERL_REF_URL,"husserl")
model=ChatterboxTTS.from_pretrained(device="cpu")
master=AudioSegment.silent(duration=500,frame_rate=24000)
for i,(sp,text) in enumerate(TURNS,1):
    ref=sref if sp=="SOCRATES" else href
    ex=0.57 if sp=="SOCRATES" else 0.42
    cfg=0.34 if sp=="SOCRATES" else 0.46
    print(i,sp,text[:70],flush=True)
    with torch.inference_mode():
        wav=model.generate(text,audio_prompt_path=str(ref),exaggeration=ex,cfg_weight=cfg)
    p=OUT/f"a{i:02d}.wav"; ta.save(str(p),wav.cpu(),model.sr)
    seg=AudioSegment.from_wav(p)
    if seg.dBFS != float("-inf"):
        seg=seg.apply_gain((-20.0 if sp=="SOCRATES" else -19.5)-seg.dBFS)
    seg=effects.compress_dynamic_range(seg,threshold=-19,ratio=1.7,attack=7,release=85)
    master+=seg+AudioSegment.silent(duration=650 if text.endswith("?") else 450,frame_rate=seg.frame_rate)

wav=OUT/"AUDITION_ONLY.wav"
mp3=OUT/"AUDITION_ONLY.mp3"
master.export(wav,format="wav")
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav),
                "-af","highpass=f=65,lowpass=f=14500,equalizer=f=3200:t=q:w=1.2:g=1.2,loudnorm=I=-16:TP=-1.2:LRA=9",
                "-ar","44100","-ac","2","-codec:a","libmp3lame","-b:a","192k",str(mp3)],check=True)
print("DONE",mp3,mp3.stat().st_size,flush=True)
