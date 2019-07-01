# import necessary package
from imutils.video import VideoStream, FPS, FileVideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import sys
from pyimagesearch.centroidtracker import CentroidTracker
# from skimage import measure

""" 
	Finding distance
"""
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)

	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 100.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0

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

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()

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
		vs = FileVideoStream('pedestrians_crossing.mp4').start()
else:
	print('Need video or cam open')
	sys.exit()
time.sleep(2.0)
fps = FPS().start()
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 600 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=600)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	rects = []
	person_list = []
	person_distance = []
	ids = []
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
			# draw the prediction on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			if CLASSES[idx] == "person":
				rects.append(box.astype("int"))
				person = "{}".format(CLASSES[idx])
				person_list.append(person)
				if confidence >= 0.5:
					mid_x = (startX+endX)/2
					mid_y = (startY+endY)/2
					apx_distance = round(((100 - (endX - startX))), 1)
					cv2.putText(frame, '{}m'.format(apx_distance/10), (int(mid_x),int(mid_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
					person_distance.append(apx_distance)

			if len(person_list) == 0:
				counted_person ="{}:".format("person")
				# cv2.putText(frame, counted_person, (10, h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
			else:
				counted_person ="{}: {}".format("person", len(person_list))
				# cv2.putText(frame, counted_person, (10, h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) 
				cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
			
	# update our centroid tracker using the computed set of bounding
	# box rectangles
	print(person_distance)
	objects = ct.update(rects)
	# loop over the tracked objects
	for (objectID, centroid) in objects.items():
		# draw both the ID of the object and the centroid of the
		# object on the output frame
		ids.append(objectID)
		# cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		# cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
	status = "person: {}".format(len(ids))
	cv2.putText(frame, status, (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
	ids = []
	# show the output frame
	cv2.imshow("Frame", frame)
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