
import cv2
import dlib
import numpy as np
import os
import sys

from imutils.video import VideoStream
import datetime
import argparse
import imutils
from imutils import face_utils
import time
import dlib
import cv2

class TooManyFaces(Exception):

    pass


class NoFace(Exception):

    pass


class Faceswapper():
    # '''
    # class of face swap
    #loading head resource
    # '''
    def __init__(self,heads_list=[],predictor_path="shape_predictor_68_face_landmarks.dat"):
        # '''
        # head_list:

        #
        # predictor_path:


        self.PREDICTOR_PATH = predictor_path
        self.FACE_POINTS = list(range(17, 68))
        self.MOUTH_POINTS = list(range(48, 61))
        self.RIGHT_BROW_POINTS = list(range(17, 22))
        self.LEFT_BROW_POINTS = list(range(22, 27))
        self.RIGHT_EYE_POINTS = list(range(36, 42))
        self.LEFT_EYE_POINTS = list(range(42, 48))
        self.NOSE_POINTS = list(range(27, 35))
        self.JAW_POINTS = list(range(0, 17))


        #face point recording
        self.ALIGN_POINTS = (self.LEFT_BROW_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_EYE_POINTS +
                                       self.RIGHT_BROW_POINTS + self.NOSE_POINTS + self.MOUTH_POINTS)


        #second face feature poing match with the first
        self.OVERLAY_POINTS = [self.LEFT_EYE_POINTS + self.RIGHT_EYE_POINTS + self.LEFT_BROW_POINTS + self.RIGHT_BROW_POINTS,
            self.NOSE_POINTS + self.MOUTH_POINTS]


        # colour fix
        self.COLOUR_CORRECT_BLUR_FRAC = 0.6


        #dlib feature point catch
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.PREDICTOR_PATH)

        #head resources
        self.heads={}
        if heads_list:
            self.load_heads(heads_list)

    def load_heads(self,heads_list):
        # '''

        # '''
        self.heads.update({os.path.split(name)[-1]:(self.read_and_mark(name)) for name in heads_list})

    def get_landmarks(self,im,fname,n=1):
        # '''

        # face match with feature point catch, raise error if two many faces or no face
        # im:
        #     numpy of img
        # fname:
        #     str of img
        #return the x,y of feature point
        # '''
        rects = self.detector(im, 1)

        if len(rects) > n:
            raise TooManyFaces('No face in '+fname)
        if len(rects) < 0:
            raise NoFace('Too many faces in '+fname)
        return np.matrix([[p.x, p.y] for p in self.predictor(im, rects[0]).parts()])

    def read_im(self,fname,scale=1):
        # '''
        # img loading
        # '''
        im = cv2.imread(fname, cv2.IMREAD_COLOR)
        if type(im)==type(None):
            print(fname)
            raise ValueError('Opencv read image {} error, got None'.format(fname))
        return im

    def read_and_mark(self,fname):
        im=self.read_im(fname)
        return im,self.get_landmarks(im,fname)

    def resize(self,im_head,landmarks_head,im_face,landmarks_face):
        # '''
        #resize size and ratio to imporve swap quality
        # '''
        scale=np.sqrt((im_head.shape[0]*im_head.shape[1])/(im_face.shape[0]*im_face.shape[1]))
        if scale>1:
            im_head=cv2.resize(im_head,(int(im_head.shape[1]/scale),int(im_head.shape[0]/scale)))
            landmarks_head=(landmarks_head/scale).astype(landmarks_head.dtype)
        else:
            pass
            # im_face=cv2.resize(im_face,(int(im_face.shape[1]*scale),int(im_face.shape[0]*scale)))
            # landmarks_face=(landmarks_face*scale).astype(landmarks_face.dtype)
        return im_head,landmarks_head,im_face,landmarks_face

    def draw_convex_hull(self,im, points, color):
        # '''
        # draw convex
        # '''
        points = cv2.convexHull(points)
        cv2.fillConvexPoly(im, points, color=color)

    def get_face_mask(self,im, landmarks,ksize=(11,11)):
        # '''
        # face mask
        # '''
        mask = np.zeros(im.shape[:2], dtype=np.float64)

        for group in self.OVERLAY_POINTS:
            self.draw_convex_hull(mask,
                             landmarks[group],
                             color=1)

        mask = np.array([mask, mask, mask]).transpose((1, 2, 0))

        mask = (cv2.GaussianBlur(mask, ksize, 0) > 0) * 1.0
        mask = cv2.GaussianBlur(mask, ksize, 0)

        return mask

    def transformation_from_points(self,points1, points2):
        # """
        # Return an affine transformation [s * R | T] such that:
        #
        #     sum ||s*R*p1,i + T - p2,i||^2
        #
        # is minimized.
        # matrix calculate
        # """
        # Solve the procrustes problem by subtracting dcentroids, scaling by the
        # standard deviation, and then using the SVD to calculate the rotation. See
        # the following for more details:
        #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2

        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= s1
        points2 /= s2

        U, S, Vt = np.linalg.svd(points1.T * points2)

        # The R we seek is in fact the transpose of the one given by U * Vt. This
        # is because the above formulation assumes the matrix goes on the right
        # (with row vectors) where as our solution requires the matrix to be on the
        # left (with column vectors).
        R = (U * Vt).T

        return np.vstack([np.hstack(((s2 / s1) * R,
                                           c2.T - (s2 / s1) * R * c1.T)),
                             np.matrix([0., 0., 1.])])

    def warp_im(self,im, M, dshape):
        # '''
        # '''
        output_im = np.zeros(dshape, dtype=im.dtype)
        cv2.warpAffine(im,
                       M[:2],
                       (dshape[1], dshape[0]),
                       dst=output_im,
                       borderMode=cv2.BORDER_TRANSPARENT,
                       flags=cv2.WARP_INVERSE_MAP)
        return output_im

    def correct_colours(self,im1, im2, landmarks_head):
        # '''
        # correct colour
        # '''
        blur_amount = int(self.COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
                                  np.mean(landmarks_head[self.LEFT_EYE_POINTS], axis=0) -
                                  np.mean(landmarks_head[self.RIGHT_EYE_POINTS], axis=0)))
        if blur_amount % 2 == 0:
            blur_amount += 1
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
        im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
        return im2.astype(np.float64) *im1_blur.astype(np.float64) /im2_blur.astype(np.float64)

    def swap(self,im1,landmarks1,im2,landmarks2):
        # '''
        #head_name
        #     head path
        # face_path:
        #     face path
        # '''
        #im2, landmarks2 = self.read_and_mark(face_path)
        #im1, landmarks1 = self.read_and_mark(head_name)
        #im_head,landmarks_head,im_face,landmarks_face=self.resize(im2,landmarks2,im1,landmarks1)
        im_head, landmarks_head, im_face, landmarks_face= im2,landmarks2,im1,landmarks1
        M = self.transformation_from_points(landmarks_head[self.ALIGN_POINTS],
                                       landmarks_face[self.ALIGN_POINTS])

        face_mask = self.get_face_mask(im_face, landmarks_face)
        warped_mask = self.warp_im(face_mask, M, im_head.shape)
        combined_mask = np.max([self.get_face_mask(im_head, landmarks_head), warped_mask],
                                  axis=0)

        warped_face = self.warp_im(im_face, M, im_head.shape)
        warped_corrected_im2 = self.correct_colours(im_head, warped_face, landmarks_head)

        out=im_head * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
        #cv2.imwrite('11.jpg',warped_corrected_im2 * combined_mask)
        #out = warped_corrected_im2
        return out

    def save(self,output_path,output_im):

        cv2.imwrite(output_path, output_im)

if __name__=='__main__':
    # '''

    # head,face_path,out=sys.argv[1],sys.argv[2],(sys.argv[3] if len(sys.argv)>=4 else 'output.jpg')
    #head, face_path, out =  '1.jpg', 'T.jpg', (sys.argv[3] if len(sys.argv) >= 4 else 'develop_output.jpg')
    swapper=Faceswapper(['1.jpg'])
    face,face_landmasks = swapper.read_and_mark('1.jpg')
    print(face_landmasks)
    #face,face_landmasks = swapper.read_and_mark('T.jpg')
    vs = VideoStream().start()

    face_landmasks = swapper.get_landmarks(face, 'face')
    #output_im = swapper.swap(head, head_landmasks, face1, face_landmasks)
    #---
    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # #----
    while 1:
        frame = vs.read()
        frame = imutils.resize(frame, width=800)
        head = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        head_landmasks = swapper.get_landmarks(head,'head')
        output_im=swapper.swap(face,face_landmasks,frame,head_landmasks)
        # #swapper.save(out,output_im)
        #------test
        # gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(gray,1.3,5)
        # x,y,w,h = faces
        #
        # #-----
        output_im[output_im>254.9]=254.9
        cv2.imshow('',output_im.astype('uint8'))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break