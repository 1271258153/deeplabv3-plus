"""
PNG 转 JPG 工具
用法:
    python png2jpg.py                              # 默认转换 VOCdevkit/VOC2007/JPEGImages 下的 png
    python png2jpg.py --path /your/image/dir       # 指定目录
    python png2jpg.py --path /your/image/dir --out /output/dir  # 指定输出目录（不覆盖原文件）
"""
import os
import argparse
from tqdm import tqdm
from PIL import Image


def png2jpg(img_dir, out_dir=None):
    """将目录下所有 png 图片转换为 jpg 图片"""
    if out_dir is None:
        out_dir = img_dir
    os.makedirs(out_dir, exist_ok=True)

    png_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.png')]
    if not png_files:
        print(f"在 {img_dir} 下未找到 png 文件")
        return

    print(f"共找到 {len(png_files)} 张 png 图片，开始转换...")
    for filename in tqdm(png_files):
        src = os.path.join(img_dir, filename)
        dst = os.path.join(out_dir, os.path.splitext(filename)[0] + '.jpg')
        img = Image.open(src)
        img = img.convert('RGB')  # PNG 可能有 RGBA 通道，转为 RGB 以兼容 JPG
        img.save(dst, 'JPEG', quality=95)
        # 转换成功后删除原 png
        if out_dir == img_dir and os.path.abspath(src) != os.path.abspath(dst):
            os.remove(src)

    print(f"转换完成！输出目录: {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PNG 转 JPG 工具')
    parser.add_argument('--path', type=str,
                        default='VOCdevkit/VOC2007/JPEGImages',
                        help='PNG 图片所在目录（默认: VOCdevkit/VOC2007/JPEGImages）')
    parser.add_argument('--out', type=str, default=None,
                        help='输出目录，默认与原路径相同（覆盖）')
    args = parser.parse_args()

    png2jpg(args.path, args.out)
