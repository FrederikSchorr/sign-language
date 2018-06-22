# import the necessary packages
import time

import numpy as np

import cv2

#from frame import resize_aspectratio

def camera_resolution(oStream, nWidth, nHeight):
	oStream.set(3, nWidth)
	oStream.set(4, nHeight)
	return


def rectangle_text(oFrame, sColor, sUpper, sLower = None, fBoxSize = 0.8):

	nHeigth, nWidth, _ = oFrame.shape
	x1 = int(nWidth * (1.0 - fBoxSize) / 2)
	y1 = int(nHeigth * (1.0 - fBoxSize) / 2)

	if sColor == "green": bgr = (84, 175, 25)
	elif sColor == "orange": bgr = (60, 125, 235)
	else: #sColor == "red": 
		bgr = (27, 13, 252)

	cv2.rectangle(oFrame, (x1, y1), (nWidth-x1, nHeigth-y1), bgr, 3)

	# display a text to the frame 
	font = cv2.FONT_HERSHEY_SIMPLEX
	fFontSize = 0.7
	textSize = cv2.getTextSize(sUpper, font, 1.0, 2)[0]
	cv2.putText(oFrame, sUpper, (x1 + 10, y1 + textSize[1] + 10), font, fFontSize, bgr, 2)	

	# 2nd text
	if (sLower != None):
		textSize = cv2.getTextSize(sLower, font, 1.0, 2)[0]
		cv2.putText(oFrame, sLower, (x1 + 10, nHeigth - y1 - 10), font, fFontSize, bgr, 2)

	return oFrame


def video_show(oStream, sColor, sUpper, sLower = None, nCountdown = 0): 
	
	if nCountdown > 0: 
		fTimeTarget = time.time() + nCountdown
	
	# loop over frames from the video file stream
	s = sUpper
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		if bGrabbed == False: continue

		if nCountdown > 0:
			fCountdown = fTimeTarget - time.time()
			s = sUpper + str(int(fCountdown)+1) + " sec"

		# paint rectangle & text, show the (mirrored) frame
		arFrame = rectangle_text(cv2.flip(arFrame, 1), sColor, s, sLower)
		cv2.imshow("Video", arFrame)
	
		# stop after countdown
		if nCountdown > 0 and fCountdown <= 0.0:
			key = -1
			break

		# Press 'q' to exit live loop
		key = cv2.waitKey(1) & 0xFF
		if key != 0xFF: break
	return key


def video_capture(oStream, sColor, sText, nTimeDuration) -> np.array:

	liFrames = []
	fTimeStart = time.time()

	# loop over frames from the video file stream
	while True:
		# grab the frame from the threaded video file stream
		(bGrabbed, arFrame) = oStream.read()
		arFrame = cv2.flip(arFrame, 1)
		liFrames.append(arFrame)

		fTimeElapsed = time.time() - fTimeStart
		s = sText + str(int(fTimeElapsed)+1) + " sec"

		# paint rectangle & text, show the frame
		arFrame = rectangle_text(arFrame, sColor, s)
		cv2.imshow("Video", arFrame)
	
		# stop after nTimeDuration sec
		if fTimeElapsed >= nTimeDuration: break

		# Press 'q' for early exit
		key = cv2.waitKey(1) & 0xFF
		if key == ord('q'): break
		cv2.waitKey(1)

	return fTimeElapsed, np.array(liFrames)


def frame_show(oStream, sColor:str, sText:str):
	""" Read frame from webcam and display it with box+text """

	(bGrabbed, oFrame) = oStream.read()
	oFrame = rectangle_text(cv2.flip(oFrame, 1), sColor, sText)
	cv2.imshow("Video", oFrame)
	cv2.waitKey(1)

	return


def unittest():
	# open a pointer to the video stream
	oStream = cv2.VideoCapture(1)
	camera_resolution(oStream, 320, 240)
	fFPS = 32.
	oStream.set(cv2.CAP_PROP_FPS, fFPS)
	#liFrames = []

	print("Launch video capture screen ...")
	# loop over action states
	sResults = ""
	while True:
		# show live video and wait for key stroke
		key = video_show(oStream, "green", "Press <blank> to start", sResults)
		
		# start!
		if key == ord(' '):
			# countdown n sec
			video_show(oStream, "orange", "Recording starts in ", None, 3)
			
			# record video for n sec
			fElapsed, liFrames = video_capture(oStream, "red", "Recording ", 5)

			# show orange wait box
			frame_show(oStream, "orange", "Translating sign ...")

			# run NN to translate video to label
			time.sleep(3)
			sResults = "Video duration {:.1f} sec, {} frames recorded, {:.1f} fps". \
				format(fElapsed, len(liFrames), len(liFrames)/fElapsed)
			print(sResults)

			# ready for next video	

		elif key == ord("+"):
			fFPS *= 2.
			print("Frame per second increased from %.1f to %.1f" % (oStream.get(cv2.CAP_PROP_FPS),fFPS))
			oStream.set(cv2.CAP_PROP_FPS, fFPS)

		elif key == ord("-"):
			fFPS /= 2.
			print("Frame per second decreased from %.1f to %.1f" % (oStream.get(cv2.CAP_PROP_FPS), fFPS))
			oStream.set(cv2.CAP_PROP_FPS, fFPS)

		# quit
		elif key == ord('q'):
			break

		cv2.waitKey(1)

	# do a bit of cleanup
	oStream.release()
	cv2.destroyAllWindows()

	return


if __name__ == '__main__':
    unittest()