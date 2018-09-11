import cv2
import dlib
import numpy as np
from skimage import transform, data

#face_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\haarcascades\haarcascade_frontalface_default.xml')
face_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\lbpcascades\lbpcascade_frontalface.xml')
eyes_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\haarcascades\haarcascade_mcs_eyepair_big.xml')
nose_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\haarcascades\haarcascade_mcs_nose.xml')
mouth_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\haarcascades\haarcascade_mcs_mouth.xml')
eye_patterns = cv2.CascadeClassifier('D:\OPENCV\sources\data\haarcascades\haarcascade_eye.xml')

mask_scale_matrix = (164, 260, 82, 208)

# sample_image = cv2.imread('T.jpg')
# face1 =sample_image
#
# # faces = face_patterns.detectMultiScale(sample_image,scaleFactor=1.1,minNeighbors=5,minSize=(100, 100))
# # print(faces)
# # for (x, y, w, h) in faces:
# #     cv2.rectangle(sample_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
# # face1 = sample_image[y:y+h,x:x+w]
# # cv2.imshow("face1",face1)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
#
# while 1:
#     face = face_patterns.detectMultiScale(face1,scaleFactor=1.1,minNeighbors=5)
#     print(face)
#     x, y, w, h=face[0]
#     face1 = face1[y:y+h,x:x+w]
#     cv2.imshow("face2",face1)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# #cv2.imwrite('face_catch.png', sample_image);

# detector = dlib.get_frontal_face_detector()
# landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
def convex(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    return img

# def mask(img,eyes_pair):
#     a,b,c,d = eyes_pair[0]
#     middle_point = [a + c//2, b + d//2]
#     im = np.zeros(img.shape[:2], dtype=np.float64)
#
#     LEFT_EYE = [a,b]
#     RIGHT_EYE = [a+c,b]
#     NOSE_LEFT = [a,b+c//3]
#     NOSE_RIGHT = [a+c,b+c//3]
#     MOUTH_LEFT = [a+c//4,b+c//3*2]
#     MOUTH_RIGHT = [a+c//4*3,b+c//3*2]
#     MOUTH_BOTTOM = [a+c//2,b+c]
#     point1 = np.array([LEFT_EYE, RIGHT_EYE, NOSE_LEFT, NOSE_RIGHT])
#     point2 = np.array([NOSE_LEFT, NOSE_RIGHT, MOUTH_RIGHT, MOUTH_BOTTOM, MOUTH_LEFT])
#     #points = cv2.convexHull(POINTS)
#     cv2.fillConvexPoly(im, point1, color=1)
#     cv2.fillConvexPoly(im, point2, color=1)
#     im = np.array([im, im, im]).transpose((1, 2, 0))
#
#     im = (cv2.GaussianBlur(im, (11, 11), 0) > 0) * 1.0
#     im = cv2.GaussianBlur(im, (11, 11), 0)
#     print(eyes_pair)
#     return im

def mask(img,eyes_pair,face2):
    a,b,c,d = eyes_pair[0]
    middle_point = [a + c//2, b + d//2]
    im = np.zeros(img.shape[:2],dtype=np.float64)
    LEFT_EYE = [a,b]
    RIGHT_EYE = [a+c,b]
    NOSE_LEFT = [a,b+c//3]
    NOSE_RIGHT = [a+c,b+c//3]
    MOUTH_LEFT = [a+c//4,b+c//3*2]
    MOUTH_RIGHT = [a+c//4*3,b+c//3*2]
    MOUTH_BOTTOM = [a+c//2,b+c]
    #point1 = np.array([LEFT_EYE, RIGHT_EYE, NOSE_LEFT, NOSE_RIGHT])
    point2 = np.array([LEFT_EYE, RIGHT_EYE, MOUTH_RIGHT, MOUTH_BOTTOM, MOUTH_LEFT])
    #points = cv2.convexHull(POINTS)
    #cv2.fillConvexPoly(im, point1, color=1)
    cv2.fillConvexPoly(im, point2, color=1)

    #cv2.ellipse(im,np.array(middle_point),(2*c,c),0,0)
    face2_gray = cv2.cvtColor(face2,cv2.COLOR_BGR2GRAY)
    ret, thresh2 = cv2.threshold(face2_gray, 100, 255, cv2.THRESH_BINARY_INV)
    #im = np.array([im, im, im]).transpose((1, 2, 0))
    im = np.array([thresh2, thresh2, thresh2]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (11, 11), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (11, 11), 0)
    return im

def rescale(eyes, mouth):
    print(eyes[0])
    eye_x, eye_y, eye_w, eye_h = eyes[0]
    mouth_x, mouth_y, mouth_w, mouth_h = mouth[0]
    height_ratio = float((mouth_y + mouth_h/2) - (eye_y + eye_h/2)) / float(mask_scale_matrix[1] - mask_scale_matrix[0])
    width_ratio = float(eye_w) / float(mask_scale_matrix[3] - mask_scale_matrix[2])
    return height_ratio, width_ratio

cap  = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_patterns.detectMultiScale(gray,1.3,5)
    print(faces)

    face2 = cv2.imread('1.jpg')

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        catch_face = img[y:y+h, x:x+w]
        eyes_pair = eyes_patterns.detectMultiScale(catch_face)
        nose = nose_patterns.detectMultiScale(catch_face)
        mouth = mouth_patterns.detectMultiScale(catch_face)

        h_r, w_r = rescale(eyes_pair, mouth)
        print(h, w)
        print(h_r, w_r)
        print(int(h * h_r), int(w * w_r))
        face2 = cv2.resize(face2,  (abs(int(w * w_r)),abs(int(h * h_r))))

        # mask1 = mask(catch_face, eyes_pair, face2)
        # img[y:y+h, x:x+w] = catch_face*(1.0-mask1)+face2*mask1
        img[y:y + abs(int(h * h_r)), x:x + abs(int(w * w_r))] = face2
        #print(mouth)
        # for (a,b,c,d) in eyes_pair:
        #     print(a,b,c,d)
        #     cv2.rectangle(catch_face,(a,b),(a+c,b+d),(255,0,0),2)
        # for (a, b, c, d) in nose:
        #     cv2.rectangle(catch_face, (a, b), (a + c, b + d), (255, 0, 0), 2)

        # img[y:y+h,x:x+w] = face2
    # faces = detector(img, 1)
    # if (len(faces) > 0):
    #     for k, d in enumerate(faces):
    #         cv2.rectangle(img, (d.left(), d.top()), (d.right(), d.bottom()), (255, 255, 255))
    #         shape = landmark_predictor(img, d)
    #         for i in range(68):
    #             cv2.circle(img, (shape.part(i).x, shape.part(i).y), 5, (0, 255, 0), -1, 8)
    #             cv2.putText(img, str(i), (shape.part(i).x, shape.part(i).y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                         (255, 2555, 255))

    # cv2.imshow("face", face2)
    cv2.imshow("face", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break