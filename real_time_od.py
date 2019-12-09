# import necessary package
from imutils.video import VideoStream, FPS, FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
from curved_lane_detection import CurvedLaneDetection
import psutil
import os
# from lane_detection import *

# construct an argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-train model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
ap.add_argument("-r", "--realtimecamera", help="real time camera ")
ap.add_argument("-v", "--video", help="path to inputed video")
args = vars(ap.parse_args())

""" 
	Finding distance
"""
def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 10 meters
DEFAULT_DISTANCE = 100.0
DEFAULT_OBJECT_WIDTH = 40.0
pxl_width = []
apx_distance = None

"""
	Init Curved Lane Detector
"""
curved = CurvedLaneDetection()
pts = np.float32([(0.27,0.5), (0.72,0.5), (-0.98,1), (1.2,1)])
pts_dst = np.float32([(0,0), (1, 0), (0,1), (1,1)])
"""
	Detect Objects
"""

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

# load serialized model
print("[INFO] loading model .. ")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args['model'])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream")
if args['realtimecamera'] or args['video']:
	if (args['realtimecamera']):
		vs = VideoStream(src=1).start()
	elif (args['video']):
		vs = FileVideoStream(args['video']).start()
else:
	print('Need video or cam open')
	sys.exit()
time.sleep(1.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=720)
	
	(h, w) = frame.shape[:2]

	# lane_detection.py => do_canny(frame) => do_segment(canny)
	# canny = do_canny(frame)
	# segment= do_segment(canny)
	# # do some hough
	# hough = cv2.HoughLinesP(canny, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
	# lines = calculate_lines(frame, hough)
	# lines_visualize = visualize_lines(frame, lines)
	# cv2.imshow("Canny", canny)
	# cv2.imshow("Hough", lines_visualize)
	# grab the frame dimensions and convert it to a blob
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	rects = []
	person_list = []
	# person_distance = []
	sat = curved.saturate(frame)
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	# contrast = cv2.addWeighted(rgb, 1.5, np.zeros(rgb.shape, rgb.dtype), 0.5, 5)
	# pipe = curved.pipeline(contrast)
	# dst = curved.perspective_warp(pipe, dst_size=(w, h), src=pts, dst=pts_dst)
	# out_img, curves, lanes, ploty = curved.sliding_window(dst)
	# curverad = curved.get_curve(contrast, curves[0],curves[1])
	# img_ = curved.draw_lanes(frame, curves[0], curves[1], dst=pts)
    # loop over the detections
	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2] 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence		
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			pxl_width.append((endX - startX))
			# Focal Length for measuring object distance
			# F = (P x D) / W
			# F = Focal lenght; P = Object width in pixels; D = Default distance; W = Object real width
			focalLength = (pxl_width[0] * DEFAULT_DISTANCE) / DEFAULT_OBJECT_WIDTH
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			if CLASSES[idx] in ("person", "car"):
				rects.append(box.astype("int"))
				person = "{}".format(CLASSES[idx])
				person_list.append(person)
				# if confidence >= 0.5:
				mid_x = (startX+endX)/2
				mid_y = (startY+endY)/2
				# apx_distance = round(((100 - (endX - startX))), 2)
				# cv2.putText(frame, '{}m'.format(apx_distance/10), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
				# apx_distance = round(((1 - ((endX - startX)/w))), 2)
				# cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
				
				# D' =(W x F) / P for measuring object distance
				apx_distance = round((distance_to_camera(DEFAULT_OBJECT_WIDTH, focalLength, (endX - startX)) / 100), 2)

				# [(0.30,0.60), (0.65,0.60), (0,1), (1,1)]
					
				if apx_distance <= (h * (1 - pts[0][1])):
					if (startX >= (w * pts[0][0]) and endX <= (w * pts[1][0])) or (startY >= (h * pts[0][1]) and endY <= (h * pts[1][1])):
						color = (0, 0, 255)  
					else:
						color = (0, 255, 0)
				else:
					color = (0, 255, 0)


				# if apx_distance is None or apx_distance >= 13:
				# 	color = (50, 255, 50)
				# elif apx_distance <= 3:
				# 	color = (50, 50, 255)
				# elif apx_distance <= 8:
				# 	color = (50, 255, 255)

				cv2.putText(rgb, '{}m'.format(apx_distance), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
					
			if len(person_list) == 0:
				counted_person ="{}:".format("person")
			else:
				if apx_distance <= (h * pts[0][1]):
					if (startX >= (w * pts[0][0]) and endX <= (w * pts[1][0])) or (startY >= (h * pts[0][1]) and endY <= (h * pts[1][1])):
						os.system('play -nq -t alsa synth 0.1 sine 700')
						color = (0, 0, 255)  
				else:
					color = (0, 255, 0)
				# if apx_distance is None or apx_distance >= 13:
				# 	color = (50, 255, 50)
				# elif apx_distance <= 3:
				# 	color = (50, 50, 255)
				# elif apx_distance <= 8:
				# 	color = (50, 255, 255)
				
				# counted_person ="{}: {}".format("person", len(person_list))
				cv2.rectangle(rgb, (startX, startY), (endX, endY), color, 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(rgb, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			
	status = "person: {}".format(len(person_list))
	cv2.putText(rgb, status, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	# show the output frame
	
	# q = curved.undistort(frame)

	# cv2.imshow("rgb", rgb)
	# cv2.imshow("Sliding", out_img)
	# cv2.imshow("Pipeline", rgb)
	# cv2.imshow("Frame", rgb)
	cv2.imshow("Result", rgb)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
	
	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
# gives a single float value
print("[INFO] cpu usage percentage: {}".format(psutil.cpu_percent()))
# gives an object with many fields
print("[INFO] total memory: {}".format(psutil.virtual_memory()[0]))
print("[INFO] available memory: {}".format(psutil.virtual_memory()[1]))
print("[INFO] persentage memory: {}".format(psutil.virtual_memory()[2]))
print("[INFO] used memory: {}".format(psutil.virtual_memory()[3]))
print("[INFO] free memory: {}".format(psutil.virtual_memory()[4]))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()