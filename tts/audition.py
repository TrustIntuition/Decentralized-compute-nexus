from pathlib import Path
import requests, subprocess, shutil, json, os
from pydub import AudioSegment, effects
from gradio_client import Client, handle_file

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

sref=OUT/"socrates_ref.wav"; href=OUT/"husserl_ref.wav"
dl(SOCRATES_REF_URL,sref); dl(HUSSERL_REF_URL,href)

client=Client("ResembleAI/Chatterbox")
try:
    api=client.view_api(return_format="dict")
    print(json.dumps(api,default=str)[:8000],flush=True)
except Exception as e:
    print("view_api warning",repr(e),flush=True)

master=AudioSegment.silent(duration=400,frame_rate=24000)

def call(text,ref,ex,cfg):
    kwargs=dict(
        text_input=text,
        audio_prompt_path_input=handle_file(str(ref)),
        exaggeration_input=ex,
        temperature_input=0.72,
        seed_num_input=37,
        cfgw_input=cfg,
        vad_trim_input=True,
    )
    try:
        return client.predict(**kwargs, api_name="/generate_tts_audio")
    except Exception as e1:
        print("named endpoint failed",repr(e1),flush=True)
        # Current official Space has a single Generate event; fn_index=0 is fallback.
        return client.predict(
            text, handle_file(str(ref)), ex, 0.72, 37, cfg, True, fn_index=0
        )

for i,(sp,text) in enumerate(TURNS,1):
    ref=sref if sp=="SOCRATES" else href
    ex=0.60 if sp=="SOCRATES" else 0.40
    cfg=0.32 if sp=="SOCRATES" else 0.48
    print("GENERATE",i,sp,text,flush=True)
    result=call(text,ref,ex,cfg)
    print("RESULT",repr(result),flush=True)

    # gradio_client may return a filepath, FileData-like path, or (sr, ndarray)-derived temp file.
    path=None
    if isinstance(result,str):
        path=result
    elif hasattr(result,"path"):
        path=result.path
    elif isinstance(result,(list,tuple)):
        for v in result:
            if isinstance(v,str) and Path(v).exists():
                path=v; break
            if hasattr(v,"path") and Path(v.path).exists():
                path=v.path; break
    if not path:
        raise RuntimeError(f"Could not resolve generated audio path: {result!r}")

    seg=AudioSegment.from_file(path)
    if seg.dBFS != float("-inf"):
        seg=seg.apply_gain((-20.0 if sp=="SOCRATES" else -19.5)-seg.dBFS)
    seg=effects.compress_dynamic_range(seg,threshold=-19,ratio=1.55,attack=7,release=90)
    master+=seg+AudioSegment.silent(duration=700,frame_rate=seg.frame_rate)

wav=OUT/"AUDITION_ONLY.wav"; mp3=OUT/"AUDITION_ONLY.mp3"
master.export(wav,format="wav")
subprocess.run(["ffmpeg","-y","-loglevel","error","-i",str(wav),
                "-af","highpass=f=65,lowpass=f=14500,equalizer=f=3200:t=q:w=1.2:g=1.0,loudnorm=I=-16:TP=-1.2:LRA=9",
                "-ar","44100","-ac","2","-codec:a","libmp3lame","-b:a","192k",str(mp3)],check=True)
print("DONE",mp3,mp3.stat().st_size,flush=True)
