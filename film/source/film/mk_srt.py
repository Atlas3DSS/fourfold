import re,sys,json,os
sys.argv=['x','a','b']
src=open('compose.py').read()
# reuse the compositor's caption timing without importing PIL work
ns={}
ns['TL']=json.load(open('../timeline.json'));ns['re']=re;ns['sys']=sys;ns['os']=os;ns['__file__']=os.path.abspath('compose.py');exec(src[src.index('MAX = 58'):src.index('LABEL_FROM')],ns)
caps=ns['CAPS']
def t(x):
    h=int(x//3600);m=int(x%3600//60);s=x%60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s%1)*1000):03d}"
open('../out/Pass_It_Up.srt','w').write("".join(f"{i+1}\n{t(a)} --> {t(b)}\n{c}\n\n" for i,(a,b,c) in enumerate(caps)))
print(len(caps))
