"""Create contact sheets for visual inspection of the actual rebuilt PDFs."""
from pathlib import Path
import argparse
import pymupdf
from PIL import Image, ImageDraw

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    files=sorted(args.directory.glob('*Figure_*.pdf'))
    for start in range(0,len(files),6):
        sheet=Image.new('RGB',(1500,1440),'#eeeeee')
        draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(files[start:start+6]):
            with pymupdf.open(path) as doc:
                pix=doc[0].get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
            im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
            im.thumbnail((480,680))
            x=(i%3)*500;y=(i//3)*720
            sheet.paste(im,(x+(500-im.width)//2,y+30))
            draw.text((x+10,y+8),path.stem,fill='black')
        sheet.save(args.directory/f'contact_{start//6+1}.png')

if __name__=='__main__':main()
