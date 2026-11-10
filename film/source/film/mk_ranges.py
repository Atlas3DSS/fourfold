# Writes film/ranges.json from timeline.json and the shot order. No Blender needed.
import json,re,os
d=os.path.dirname(os.path.abspath(__file__))
t=json.load(open(d+'/../timeline.json'))
src=open(d+'/shots.py').read()
order=eval(src[src.index('ORDER = ')+8:])
fr=lambda x:1+round(x*24)
R={}
for k,(s,c) in enumerate(order):
    a=fr(t[c]['start'])
    b=fr(t[order[k+1][1]]['start']) if k+1<len(order) else fr(t['_total'])
    R[s]=(a,b)
json.dump(R,open(d+'/ranges.json','w'))
for s,(a,b) in R.items(): print(s,a,b,f"{(a-1)/24:.1f}-{(b-1)/24:.1f}s")
print('unique frames',sum((b-a+1)//2 for a,b in R.values()))
