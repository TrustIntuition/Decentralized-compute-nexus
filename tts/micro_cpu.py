from pathlib import Path
import requests, subprocess, torch, torchaudio as ta
from pydub import AudioSegment, effects
from chatterbox.tts import ChatterboxTTS

ROOT=Path(__file__).parent
OUT=ROOT/"output"; OUT.mkdir(parents=True,exist_ok=True)
SURL="https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/48402e0ec040427bb1848400600ea74e/id=4dc9fda8-4e3c-4d1f-8ddb-1fba283f4c5e.wav"
HURL="https://resource2.heygen.ai/text_to_speech/7cb0a2dbb3cc4537b4c2bdc736b04dc6/0e2ff5b962084420879e076a2345d13f/id=4b038f34-545a-4bfd-b13b-4c3e7ee3ce09.wav"
TURNS=[
("S","Then purge the picture. No darkness. No silence. No waiting. No empty duration. No sleeper. No observer whispering, I am gone. What remains?"),
("H","No further experience. And no, that no-further-experience is never given, retained, remembered, fulfilled, or encountered by a later phase.")
]
def dl(u,p):
    r=requests.get(u,timeout=120); r.raise_for_status(); p.write_bytes(r.content)
s=OUT/"s.wav"; h=OUT/"h.wav"; dl(SURL,s); dl(HURL,h)
torch.set_num_threads(4)
m=ChatterboxTTS.from_pretrained(device="cpu")
master=AudioSegment.silent(duration=300,frame_rate=24000)
for i,(sp,t) in enumerate(TURNS):
    with torch.inference_mode():
        w=m.generate(t,audio_prompt_path=str(s if sp=="S" else h),
                     exaggeration=0.60 if sp=="S" else 0.40,
                     cfg_weight=0.32 if sp=="S" else 0.48)
    p=OUT/f"x{i}.wav"; ta.save(str(p),w.cpu(),m.sr)
    seg=AudioSegment.from_wav(p)
    if seg.dBFS != float("-inf"):
        seg=seg.apply_gain((-20.0 if sp=="S" else -19.5)-seg.dBFS)
    seg=effects.compress_dynamic_range(seg,threshold=-19,ratio=1.55,attack=7,release=90)
    master+=seg+AudioSegment.silent(duration=700,frame_rate=seg.frame_rate)
wav=OUT/"CPU_MICRO_AUDITION.wav"; mp3=OUT/"CPU_MICRO_AUDITION.mp3"
master.export(wav,format="wav")
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav),
 "-af","highpass=f=65,lowpass=f=14500,equalizer=f=3200:t=q:w=1.2:g=1.0,loudnorm=I=-16:TP=-1.2:LRA=9",
 "-ar","44100","-ac","2","-codec:a","libmp3lame","-b:a","192k",str(mp3)],check=True)
print("DONE",mp3,mp3.stat().st_size,flush=True)
