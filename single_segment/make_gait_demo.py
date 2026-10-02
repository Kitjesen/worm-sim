"""Combine the two verified 25 fps SOFA replays, preserving physical time."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

OUT=Path(__file__).resolve().parent/'gait_compare_20261002'


def main():
    readers=[imageio_ffmpeg.read_frames(str(OUT/m/'whole_sofa_studio.mp4')) for m in ['worm','snake']]
    for reader in readers:
        meta=next(reader)
        assert meta['fps']==25 and tuple(meta['size'])==(1600,900)
    writer=imageio_ffmpeg.write_frames(str(OUT/'gait_comparison.mp4'),(960,540),fps=25,
        codec='libx264',quality=8,pix_fmt_out='yuv420p',macro_block_size=2,output_params=['-movflags','+faststart'])
    writer.send(None)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    labels=['纯蠕动：35 mm 目标行程；关节目标为 0°', '纯蛇形：关节目标 15°；收绳命令固定']
    gif=[];count=0
    for count,frames in enumerate(zip(*readers),1):
        canvas=Image.new('RGB',(960,540))
        for i,raw in enumerate(frames):
            im=Image.frombytes('RGB',(1600,900),raw).crop((0,200,1600,650)).resize((960,270),Image.Resampling.LANCZOS)
            draw=ImageDraw.Draw(im)
            draw.rounded_rectangle((10,8,520,39),radius=4,fill='#f4f6f7')
            draw.text((20,10),labels[i],font=font,fill='#223b4c')
            canvas.paste(im,(0,i*270))
        writer.send(np.asarray(canvas))
        if (count-1)%2==0:gif.append(canvas.convert('P',palette=Image.Palette.ADAPTIVE,colors=192))
        if count==151:canvas.save(OUT/'gait_comparison.png')
    writer.close()
    assert count==300 and len(gif)==150
    gif[0].save(OUT/'gait_comparison.gif',save_all=True,append_images=gif[1:],duration=80,loop=0)
    check=imageio_ffmpeg.read_frames(str(OUT/'gait_comparison.mp4'));meta=next(check)
    assert sum(1 for _ in check)==300 and meta['fps']==25
    print('Comparison: 300 video frames / 150 GIF frames, both 12 s at 1x.')


if __name__=='__main__':main()
