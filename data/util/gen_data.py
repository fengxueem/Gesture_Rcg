import cv2
import imutils
import numpy as np
import math
import os, sys
from argparse import ArgumentParser
from brisque import BRISQUE
from datetime import datetime

def main(args):
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)
    # ROI coordinates
    img_w = 112
    img_h = 112
    top = 160
    left = 300
    bottom = top + img_h
    right = left + img_w

    image_num = 0

    start_recording = False
    # make a new folder storing images
    today_str = datetime.now().strftime("%y%m%d%H%M%S")
    saving_folder_path = "./" + today_str + '_' + args.file_prefix
    os.mkdir(saving_folder_path)
    saving_img_path = saving_folder_path + '/' + args.file_prefix + '_'
    print("Save images to " + saving_img_path)
    brisq = BRISQUE()
    while(True):
        # get the current frame
        (grabbed, frame) = camera.read()
        if (grabbed == True):
            # fix the frame width to be 640
            frame = imutils.resize(frame, width=640)
            # clone the frame for rendering
            clone = frame.copy()

            # get the ROI
            roi = frame[top:bottom, left:right]

            # convert the roi to grayscale and blur it
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            image_quality = brisq.get_score(gray)
            # sometimes brisq outputs nan, we should ignore these anomolies
            if math.isnan(image_quality):
                continue
            if start_recording and image_quality < float(args.ref_quality):
                file_full_path = saving_img_path + str(image_num) + ".jpg"
                cv2.imwrite(file_full_path, gray,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                # make sure the image is saved
                while(os.path.getsize(file_full_path) < 1):
                    cv2.imwrite(file_full_path, gray,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                image_num += 1
                print('%d images saved\r' % (image_num))
            
            # specify the font and draw the key using puttext
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(clone,str(int(image_quality * 10) / 10),(right - img_w, top), font, .8,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow("Gray", gray)

            # draw the segmented hand
            cv2.rectangle(clone, (right, top), (left, bottom), (0,255,0), 2)

            # display the frame with segmented hand
            cv2.imshow("Video Feed", clone)

            # observe the keypress by the user
            keypress = cv2.waitKey(1) & 0xFF

            # if the user pressed "q", then stop looping
            if keypress == ord("q") or image_num > int(args.img_num) - 1:
                break
        
            if keypress == ord("s"):
                start_recording = not start_recording

        else:
            print("[Warning!] Error input, Please check your(camra Or video)")
            break
    # free up memory
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate dataset. Press 's' to start or stop; Press 'q' to quit")
    parser.add_argument('ref_quality', help="Upper bound quality of saved images")
    parser.add_argument('file_prefix', help="Prefix of saved images")
    parser.add_argument('img_num', help="Max number of saved images")
    args = parser.parse_args()
    main(args)