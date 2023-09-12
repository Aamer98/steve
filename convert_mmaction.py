import os
import glob
import pickle
import argparse

import torchvision.utils as vutils
from torchvision import transforms

import cv2
import numpy as np


def vid_array(vid_path):

    video = cv2.VideoCapture(vid_path)
    frames = []
    while(video.isOpened()):
        ret, frame = video.read()
        if ret:
            frames.append(frame)
        else:
            break
    frames = np.array(frames)
    return frames


parser = argparse.ArgumentParser()
parser.add_argument('--source_path', type=str, default='/home/as26840@ens.ad.etsmtl.ca/data/hmdb51/videos')
parser.add_argument('--out_path', default='/home/as26840@ens.ad.etsmtl.ca/data/hmdb51_steve')
parser.add_argument('--image_size', type=int, default=128)
args   = parser.parse_args()

vid_paths = glob.glob("/home/as26840@ens.ad.etsmtl.ca/data/hmdb51/videos/*/*.avi")

to_tensor = transforms.ToTensor()
resize_tensor = transforms.Resize((args.image_size,args.image_size))

meta_data = []

b = 0
for vpath in vid_paths:

    print(f'Reading = {vpath}')
    video = vid_array(vpath)
    T, *_ = video.shape

    # setup dirs
    path_vid = os.path.join(args.out_path, f"{b:08}")
    os.makedirs(path_vid, exist_ok=True)

    # create metadata
    vid_m = vpath.split('/')
    meta_data.append({'dataset':vid_m[4], 'class':vid_m[6], 'file_name':vid_m[7],
                      'folder_path_steve':path_vid})

    for t in range(T):
        # breakpoint()
        img = video[t]
        img = to_tensor(img)
        vutils.save_image(img, os.path.join(path_vid, f"{t:08}_image.png"))

    b += 1


# save metadata
with open(f'/home/as26840@ens.ad.etsmtl.ca/data/{vid_m[4]}_steve_metadata.pkl', 'wb') as file:
    pickle.dump(meta_data, file)