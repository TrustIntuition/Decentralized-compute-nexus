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
("SOCRATES","Husserl, you have come to tell me what consciousness is. I have spent my life asking men what they mean by the words they use. So begin simply. When I see this cup, what is actually given?"),
("HUSSERL","Only a profile is strictly seen. The far side is absent, yet intended as belonging to the same cup. Consciousness is always consciousness of something, and every appearance carries a horizon of further possible appearances."),
("SOCRATES","Then seeing already contains what is absent."),
("HUSSERL","Absence, yes. Not nothingness."),
("SOCRATES","You distinguish them quickly. Suppose the whole world were doubtful. Would the appearing also be doubtful?"),
("HUSSERL","That is why I perform the epoché. I do not deny the world; I suspend its unquestioned existence. What remains is the field of appearing itself: perceiving, remembering, judging, imagining, and the objects precisely as they are given."),
("SOCRATES","So you can bracket the world, but not the fact that something appears. And to whom does it appear?"),
("HUSSERL","To transcendental subjectivity. Not Edmund Husserl as a biological man, but the ego-pole of the stream of experience.")
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
