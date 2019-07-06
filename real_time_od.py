# import necessary package
from imutils.video import VideoStream, FPS, FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
from lane_detection import *
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
	Detect Objects
"""
# construct an argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, help="path to Caffe pre-train model")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="minimum probability to filter weak detections")
ap.add_argument("-r", "--realtimecamera", help="real time camera ")
ap.add_argument("-v", "--video", help="path to inputed video")
args = vars(ap.parse_args())

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
		vs = VideoStream(src=0).start()
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
	
	# lane_detection.py => do_canny(frame) => do_segment(canny)
	canny = do_canny(frame)
	segment= do_segment(canny)
	# do some hough
	hough = cv2.HoughLinesP(canny, 2, np.pi / 180, 100, np.array([]), minLineLength = 100, maxLineGap = 50)
	lines = calculate_lines(frame, hough)
	lines_visualize = visualize_lines(frame, lines)
	cv2.imshow("Canny", canny)
	cv2.imshow("Hough", lines_visualize)
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	rects = []
	person_list = []
	# person_distance = []
	
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
			if CLASSES[idx] == "person":
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
				
				if apx_distance is None or apx_distance >= 13:
					color = (50, 255, 50)
				elif apx_distance <= 3:
					color = (50, 50, 255)
				elif apx_distance <= 8:
					color = (50, 255, 255)

				cv2.putText(frame, '{}m'.format(apx_distance), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
					
			if len(person_list) == 0:
				counted_person ="{}:".format("person")
			else:
				if apx_distance is None or apx_distance >= 13:
					color = (50, 255, 50)
				elif apx_distance <= 3:
					color = (50, 50, 255)
				elif apx_distance <= 8:
					color = (50, 255, 255)
				
				counted_person ="{}: {}".format("person", len(person_list))
				cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			
	status = "person: {}".format(len(person_list))
	cv2.putText(frame, status, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	# show the output frame
	output = cv2.addWeighted(frame, 0.9, lines_visualize, 1, 1)
	cv2.imshow("Frame", output)
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
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()