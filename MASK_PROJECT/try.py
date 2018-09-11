import cv2
import numpy as np
import dlib
import math
import time
from imutils.video import VideoStream

def fps_detect(record_time):

    fps = time.time()-record_time
    record_time = time.time()
    fps = 1/fps
    cv2.putText(frame, str(fps),
                (int(0.05 * frame.shape[1]), int(0.05 * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                1, 8)
    return record_time

record_time = time.time()

vs = VideoStream().start()
l = 0
while(1):
    if l%2 == 0:
        l += 1
        continue
    else:
        # Capture frame from camera
        # frame = vs.start().read()
        frame = vs.read()
        frame=cv2.flip(frame,1)
        record_time = fps_detect(record_time)
        cv2.imshow("Frame", frame)
        l+=1
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

cv2.destroyAllWindows()
vs.stop()