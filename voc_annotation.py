import os
import random
import re

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   训练集来源：VOCdevkit/VOC2007/JPEGImages（所有图片）
#   验证集来源：dataset/gt_png/val + dataset/label_png/val
#-------------------------------------------------------#
VOCdevkit_path      = 'VOCdevkit'
train_img_dir       = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages')
val_img_dir         = 'dataset/gt_png/val'
val_label_dir       = 'dataset/label_png/val'

if __name__ == "__main__":
    print("Generate txt in ImageSets.")
    saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    os.makedirs(saveBasePath, exist_ok=True)

    #-------------------------------------------------------#
    #   训练集：JPEGImages 里的所有图片（原图+增强）
    #   按原始图片分组，写入时同时写入原图和增强版本
    #-------------------------------------------------------#
    def get_base_name(filename):
        return re.sub(r'_aug\d+$', '', filename)

    train_files = [f for f in os.listdir(train_img_dir) if f.endswith('.jpg')]
    base_to_files = {}
    for f in train_files:
        base = get_base_name(f[:-4])
        if base not in base_to_files:
            base_to_files[base] = []
        base_to_files[base].append(f[:-4])

    # 写入 train.txt（所有训练图片，包含原图和增强）
    all_train_names = []
    for base, files in base_to_files.items():
        all_train_names.extend(files)

    with open(os.path.join(saveBasePath, 'train.txt'), 'w') as f:
        for name in sorted(all_train_names):
            f.write(name + '\n')

    #-------------------------------------------------------#
    #   验证集：dataset/gt_png/val 里的图片
    #-------------------------------------------------------#
    val_files = [f[:-4] for f in os.listdir(val_img_dir) if f.endswith('.jpg')]

    with open(os.path.join(saveBasePath, 'val.txt'), 'w') as f:
        for name in sorted(val_files):
            f.write(name + '\n')

    # trainval.txt = train + val（用于 get_miou.py 等场景）
    with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as f:
        for name in sorted(all_train_names):
            f.write(name + '\n')
        for name in sorted(val_files):
            f.write(name + '\n')

    # test.txt 为空（不单独划分测试集）
    open(os.path.join(saveBasePath, 'test.txt'), 'w').close()

    num_train = len(all_train_names)
    num_val = len(val_files)
    print(f"train size {num_train}")
    print(f"val size {num_val}")
    print("Generate txt in ImageSets done.")

    #-------------------------------------------------------#
    #   检查训练集标签格式
    #-------------------------------------------------------#
    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    segfilepath = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    total_seg = [f for f in os.listdir(segfilepath) if f.endswith('.png')]

    classes_nums = np.zeros([256], int)
    for name in tqdm(total_seg):
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片%s，请查看具体路径下文件是否存在以及后缀是否为png。" % png_file_name)

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("训练集来源: VOCdevkit/VOC2007/JPEGImages")
    print("验证集来源: dataset/gt_png/val + dataset/label_png/val")