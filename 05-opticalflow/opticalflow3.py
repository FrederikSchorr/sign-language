import os
import numpy as np
import cv2
from glob import glob
from multiprocessing import Pool


_IMAGE_SIZE = 256


def cal_for_frames(video_path):
    frames = glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0])
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        print("Calculating optical flow for frame %d" % i)
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2*bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow


def save_flow(video_flows, flow_path):
    print(os.getcwd())
    for i, flow in enumerate(video_flows):
        sPath = os.path.join(flow_path.format('u'), "{:06d}.jpg".format(i))
        retval = cv2.imwrite(sPath, flow[:, :, 0])
        print("%2d | %s | writing to %s | %d" % (i, str(flow.shape), sPath, retval))
        cv2.imwrite(os.path.join(flow_path.format('v'), "{:06d}.jpg".format(i)),
                    flow[:, :, 1])


def gen_video_path():
    path = []
    flow_path = []
    length = []
    base = ''
    flow_base = ''
    for task in ['train', 'dev', 'test']:
        videos = os.listdir(os.path.join(base, task))
        for video in videos:
            tmp_path = os.path.join(base, task, video, '1')
            tmp_flow = os.path.join(flow_base, task, '{:s}', video)
            tmp_len = len(glob(os.path.join(tmp_path, '*.png')))
            u = False
            v = False
            if os.path.exists(tmp_flow.format('u')):
                if len(glob(os.path.join(tmp_flow.format('u'), '*.jpg'))) == tmp_len:
                    u = True
            else:
                os.makedirs(tmp_flow.format('u'))
            if os.path.exists(tmp_flow.format('v')):
                if len(glob(os.path.join(tmp_flow.format('v'), '*.jpg'))) == tmp_len:
                    v = True
            else:
                os.makedirs(tmp_flow.format('v'))
            if u and v:
                print('skip:' + tmp_flow)
                continue

            path.append(tmp_path)
            flow_path.append(tmp_flow)
            length.append(tmp_len)
    return path, flow_path, length


def extract_flow(video_path, flow_path):
    flow = cal_for_frames(video_path)
    save_flow(flow, flow_path)
    print('complete:' + flow_path)
    return


if __name__ =='__main__':
    #pool = Pool(2)   # multi-processing

    #video_paths, flow_paths, video_lengths = gen_video_path()

    #pool.map(extract_flow, zip(video_paths, flow_paths))

    sFrameDir = "05-opticalflow/data/frame/M_01680"
    sFlowDir = "05-opticalflow/data/opticalflow/M_01680/{:s}"
    extract_flow(sFrameDir, sFlowDir)
