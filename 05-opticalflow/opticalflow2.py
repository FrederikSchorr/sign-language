#!/usr/bin/env python
import argparse
import cv2
import os
import glob
import sys
import numpy as np
import scipy.io as sio
import time

def cvReadGrayImg(img_path):
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

def saveOptFlowToImage(flow, basename:str, merge:bool):
    #print("Save flow to %s" % basename)
    if merge:
        # save x, y flows to r and g channels, since opencv reverses the colors
        cv2.imwrite(basename+'.png', flow[:,:,::-1])
    else:
        #cv2.imwrite(basename+'_x.jpeg', flow[...,0])
        cv2.imwrite(basename+'_y.jpeg', flow[...,1])


def main(vid_dir:str, save_dir:str):
    norm_width = 500.
    bound = 15

    images = sorted(glob.glob(os.path.join(vid_dir,'*')))
    print ("Processing {}: {} files... ".format(vid_dir, len(images))),
    sys.stdout.flush()
    tic = time.time()
    img2 = cvReadGrayImg(images[0])

    hsv = np.zeros_like(cv2.imread(images[0]))
    hsv[...,1] = 255

    # loop through all images
    for ind, img_path in enumerate(images[:-1]):
        img1 = img2
        img2 = cvReadGrayImg(images[ind+1])
        h, w = img1.shape
        fxy = norm_width / w
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy),
            flow=None, pyr_scale=0.5, levels=1, winsize=15, iterations=2,
            poly_n=5, poly_sigma=1.1, flags=0)
            #0.5, 3, 15, 3, 7, 1.5, 0)
        # map optical flow back
        flow = flow / fxy
        # normalization
        #flow = np.round((flow + bound) / (2. * bound) * 255.)
        #flow[flow < 0] = 0
        #flow[flow > 255] = 255
        flow = cv2.resize(flow, (w, h))

        # Fill third channel with zeros
        flow = np.concatenate((flow, np.zeros((h,w,1))), axis=2)

        # save
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        basename = os.path.splitext(os.path.basename(img_path))[0]
        saveOptFlowToImage(flow, os.path.join(save_dir, basename), merge=False)

        """mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(cv2.imread(img_path))
        hsv[...,1] = 255
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)"""

        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        cv2.imwrite(save_dir + "/" + "flow_" + basename + ".jpg", bgr)

        cv2.imshow("image", img2)
        cv2.imshow('optical flow',bgr)
        k = cv2.waitKey(1000) & 0xff
        if k == 27:
            break

    # duplicate last frame
    basename = os.path.splitext(os.path.basename(images[-1]))[0]
    saveOptFlowToImage(flow, os.path.join(save_dir, basename), merge=False)
    toc = time.time()
    print("{:.2f} sec, {:.2f} fps".format((toc-tic), 1. * len(images) / (toc - tic)))

    return

if __name__ == '__main__':
    """parser = argparse.ArgumentParser()
    parser.add_argument('vid_dir')
    parser.add_argument('save_dir')
    parser.add_argument('--bound', type=float, required=False, default=15,
                        help='Optical flow bounding. [-bound, bound] will be mapped to [0, 255].')
    parser.add_argument('--merge', dest='merge', action='store_true',
                        help='Merge optical flow in x and y axes into RGB images rather than saving each to a grayscale image.')
    parser.add_argument('--debug', dest='visual_debug', action='store_true',
                        help='Visual debugging.')
    parser.set_defaults(merge=False, visual_debug=False)
    args = parser.parse_args()"""

    sFrameDir = "05-opticalflow/data/frame/M_01680"
    sFlowDir = "05-opticalflow/data/opticalflow/M_01680"
    main(sFrameDir, sFlowDir)

    
