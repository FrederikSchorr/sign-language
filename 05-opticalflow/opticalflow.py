import cv2
import numpy as np


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    TVL1 = cv2.DualTVL1OpticalFlow_create(
        warps=1
    )
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    #flow = (flow + bound) * (255.0 / (2*bound))
    #flow = np.round(flow).astype(int)
    #flow[flow >= 255] = 255
    #flow[flow <= 0] = 0

    return flow


def flowToColor(flow, hsv):
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return hsv, bgr

def main():

    sVideoPath = "../datasets/04-chalearn/train/c049/M_00831.avi"
    sFlowDir = "05-opticalflow/data/opticalflow"

    cap = cv2.VideoCapture(sVideoPath)

    bRet, frame1 = cap.read()
    if bRet == False: raise ValueError("Cannot open video file")

    print("Video %s | %d frames | %.1f fps" % \
        (sVideoPath, cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)))
    
    cap.set(cv2.CAP_PROP_FPS, 60)
    print("Try to change fps: %d frames | %.1f fps" % \
        (cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)))

    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    while(1):
        ret, frame2 = cap.read()
        if ret == False: break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flowFarnback = cv2.calcOpticalFlowFarneback(prvs,next, #None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow=None, pyr_scale=0.5, levels=1, winsize=15, iterations=2,
            poly_n=5, poly_sigma=1.1, flags=0)
        _, bgrFarnback = flowToColor(flowFarnback, hsv)

        flowTVL1 = compute_TVL1(prvs, next)
        _, bgrTVL1 = flowToColor(flowTVL1, hsv)
        
        cv2.imshow("frame", frame2)
        cv2.imshow('flowFarnback',bgrFarnback)
        cv2.imshow('flowTVL1',bgrTVL1)
        
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break
        """elif k == ord('s'):
            cv2.imwrite(sFlowDir + '/opticalfb.png',frame2)
            cv2.imwrite(sFlowDir + '/opticalhsv.png',bgr)"""
        prvs = next
    cap.release()
    cv2.destroyAllWindows()

    return

if __name__ =='__main__':
    main()