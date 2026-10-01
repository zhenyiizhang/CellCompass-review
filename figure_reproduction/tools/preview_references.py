from pathlib import Path
import fitz
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[1]
out=root/'output'/'reference_preview'
out.mkdir(parents=True,exist_ok=True)
items=list((root/'artwork').glob('Figure_*/*.ai'))+list((root/'reference/si').rglob('*.pdf'))
tiles=[]
for p in items:
    doc=fitz.open(p)
    page=doc[0]
    pix=page.get_pixmap(matrix=fitz.Matrix(700/page.rect.width,700/page.rect.width),alpha=False)
    dest=out/(p.parent.name+'_'+p.stem+'.png')
    pix.save(dest)
    im=Image.open(dest).convert('RGB')
    im.thumbnail((450,300))
    tile=Image.new('RGB',(470,335),'white')
    tile.paste(im,((470-im.width)//2,28))
    ImageDraw.Draw(tile).text((5,5),p.parent.name+'/'+p.stem,fill='black')
    tiles.append(tile)
for i in range(0,len(tiles),9):
    sheet=Image.new('RGB',(1410,1005),'#dddddd')
    for j,t in enumerate(tiles[i:i+9]):sheet.paste(t,((j%3)*470,(j//3)*335))
    sheet.save(out/f'contact_{i//9}.png')
print(out)
