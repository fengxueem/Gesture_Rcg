import cv2
import os, sys
import Augmentor
import glob
import shutil
from argparse import ArgumentParser

def main(args):
    # Augmentor is really simple to use!
    aug_pipe = Augmentor.Pipeline(args.img_path)
    aug_pipe.flip_left_right(probability=0.5)
    # 30,000 images per class
    aug_pipe.sample(30000)
    # make sure that the output image path exists
    if not os.path.exists(args.output_img_path):
        os.makedirs(args.output_img_path)
    # the default output path for Augmentor
    aug_output_path = os.path.join(args.img_path, "output")
    img_num = 0
    print("Starting saving images...")
    # transverse all augmented images and move them to the output image path
    for path,dir_list,file_list in os.walk(aug_output_path):
        for file in file_list:
            # make sure it's an image
            if ".jpg" in file:
                img_num += 1
                src_path = os.path.join(path, file)
                dst_path = os.path.join(args.output_img_path, str(img_num) + ".jpg")
                shutil.move(src_path, dst_path)
                print("Rename: %d / 30000\r" % img_num, end="")
    print("Done:)")

if __name__ == '__main__':
    parser = ArgumentParser(description="Preprocess raw data and make dataset file")
    parser.add_argument('img_path', help="Path of raw image")
    parser.add_argument('output_img_path', help="Path of output images")
    args = parser.parse_args()
    main(args)
