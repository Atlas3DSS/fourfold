import sys,glob,os,json
from PIL import Image,ImageDraw,ImageFont
out,dest=sys.argv[1],sys.argv[2]; fs=sorted(glob.glob(out+'/f*.png'))
tw,th=480,270;W=4
rows=(len(fs)+W-1)//W
im=Image.new('RGB',(tw*W,th*rows),(30,30,30));d=ImageDraw.Draw(im)
for i,f in enumerate(fs):
    n=int(os.path.basename(f)[1:7]);t=Image.open(f).convert('RGB').resize((tw,th));x,y=tw*(i%W),th*(i//W);im.paste(t,(x,y));d.text((x+6,y+4),f'{n} {(n-1)/24:.1f}s',fill=(255,255,0))
im.save(dest);print(dest,len(fs))
