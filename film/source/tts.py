# Runs with right_tool/.venv; writes tts/ and timeline.json.
import json,sys,os,numpy as np,soundfile as sf
sys.path.insert(0,os.path.dirname(__file__))
from narration import LINES
from kokoro_onnx import Kokoro
d=os.path.dirname(os.path.abspath(__file__))
k=Kokoro(f"{d}/models/kokoro-v1.0.onnx",f"{d}/models/voices-v1.0.bin")
tl={};t=0.0;SR=24000
for id_,say,cap,gap in LINES:
    if say:
        a,sr=k.create(say,voice="af_heart",speed=1.12,lang="en-us"); assert sr==SR
        # trim silence
        env=np.abs(a); idx=np.where(env>0.004)[0]; a=a[max(idx[0]-240,0):min(idx[-1]+1200,len(a))]
        sf.write(f"{d}/tts/{id_}.wav",a,SR); dur=len(a)/SR
    else: dur=0.0
    tl[id_]={"start":round(t,3),"dur":round(dur,3),"end":round(t+dur,3),"cap":cap}
    t+=dur+gap
tl["_total"]=round(t,3)
json.dump(tl,open(f"{d}/timeline.json","w"),indent=1)
print("total",t); print({k:v['dur'] for k,v in tl.items() if k!='_total'})
