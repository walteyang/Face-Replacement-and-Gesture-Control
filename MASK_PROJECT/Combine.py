# import the necessary packages
from imutils.video import VideoStream
import datetime
import argparse
import imutils
from imutils import face_utils
import time
import dlib
import cv2
import numpy as np
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
# print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

vs = VideoStream().start()
#time.sleep(2.0)
record_time = time.time()
mask_index = 1
timestamp = 0
timestamp_1 = -1



def get_mask(img,face2):
    face2_gray = cv2.cvtColor(face2, cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(face2_gray, 100, 255, cv2.THRESH_BINARY_INV)
    # im = np.array([im, im, im]).transpose((1, 2, 0))
    im = np.array([thresh2, thresh2, thresh2]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (11, 11), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (11, 11), 0)
    return im

def get_face_size(feature_points):
    wide = int(math.sqrt((feature_points[16][0] - feature_points[0][0])**2+(feature_points[16][1] - feature_points[0][1])**2))
    high = int(math.sqrt((feature_points[8][1] - feature_points[27][1])**2+(feature_points[8][0] - feature_points[27][0])**2)*1.5)
    # x = feature_points[0][0]
    # y = feature_points[8][1]-high

    x = feature_points[27][0] - wide // 2
    y = feature_points[27][1] - int(high/2.5)
    angle = math.atan2(-(feature_points[16][1] - feature_points[0][1]),(feature_points[16][0] - feature_points[0][0]))
    return x,y,wide,high,math.degrees(angle)

def rotation(img,angle):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((h/2,w/2),angle,1)
    img = cv2.warpAffine(img,M,(w,h))
    return img

def fix_mask(img,middle_x,left_x,right_x):
    h,w = img.shape[:2]
    left_img = img[0:h,0:int(w/2)]
    right_img = img[0:h,int(w/2):w]
    left_img = cv2.resize(left_img,(abs(left_x - middle_x),h))
    right_img = cv2.resize(right_img,(abs(right_x - middle_x),h))
    img = np.hstack((left_img,right_img))
    return img

def hide_trigger(r):
    global mask_index
    global timestamp_1
    if len(r) == 0:
        if timestamp_1 == 0:
            mask_index += 1
            timestamp_1 = time.time()
        elif time.time() - timestamp_1 > 0.5:
            mask_index += 1
            timestamp_1 = time.time()
        else:
            return "mask" + str(mask_index) + ".jpg"
    if mask_index > 10:
        mask_index = 1
    elif mask_index < 1:
        mask_index = 10
    return "mask" + str(mask_index) + ".jpg"

def rotation_trigger(shape):
    global mask_index
    global timestamp
    if (shape[16][0]-shape[27][0]) / (shape[27][0]-shape[0][0]) > 2:
        if timestamp == 0:
            mask_index += 1
            timestamp = time.time()
        elif time.time() - timestamp > 1.5:
            mask_index += 1
            timestamp = time.time()
        else:
            return "mask" + str(mask_index) + ".jpg"
    elif((shape[27][0]-shape[0][0]) / (shape[16][0]-shape[27][0]) > 2):
        if timestamp == 0:
            mask_index += 1
            timestamp = time.time()
        elif time.time() - timestamp > 1.5:
            mask_index -= 1
            timestamp = time.time()
        else:
            return "mask" + str(mask_index) + ".jpg"
    if mask_index > 10:
        mask_index = 1
    elif mask_index < 1:
        mask_index = 10
    return "mask" + str(mask_index) + ".jpg"

def fps_detect(record_time,frame1):

    fps = time.time()-record_time
    record_time = time.time()
    fps = 1/fps
    cv2.putText(frame1, str(fps),
                (int(0.05 * frame1.shape[1]), int(0.05 * frame1.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1, 8)
    return record_time
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame1 = vs.read()
    frame1 = imutils.resize(frame1, width=200)
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    mask = hide_trigger(rects)

    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # for (x, y) in shape:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)


        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        # i =  0
        # for (x, y) in shape:
        #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        #     cv2.putText(frame, str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                 (255, 255, 255))
        #     i+=1
        #     feature_points.append((x,y))

        x,y,w,h,angle = get_face_size(shape)

        mask = rotation_trigger(shape)
        face2 = cv2.imread("mask collection/" + mask)
        face2 = cv2.resize(face2,(w,h))
        catch_face = frame1[y:y + h, x:x + w]
        background_mask = get_mask(catch_face, face2)

        background_mask = fix_mask(background_mask, shape[27][0], shape[0][0], shape[16][0])
        face2 = fix_mask(face2, shape[27][0], shape[0][0], shape[16][0])

        background_mask = rotation(background_mask,angle)
        face2 = rotation(face2,angle)

        background_mask = cv2.resize(background_mask, (w, h))
        face2 = cv2.resize(face2, (w, h))

        # cv2.imshow('mask1',mask1)
        # cv2.imshow('face2',face2)
        # cv2.waitKey(0)
        try:
            frame1[y:y + h, x:x + w] = catch_face * (1.0 - background_mask) + face2 * background_mask
        except ValueError:
            continue
    record_time = fps_detect(record_time, frame1)
    cv2.imshow("Frame", frame1)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()