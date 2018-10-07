from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True
import argparse

import os
import cv2
import dlib
import numpy as np
from mtcnn.mtcnn import MTCNN



import face_recognition
from matlab_cp2tform import get_similarity_transform_for_cv2
from PIL import Image
from get_landmarks import get_five_points_landmarks
import net_sphere

def alignment(src_img,src_pts):
    ref_pts = [ [30.2946, 51.6963],[65.5318, 51.5014],
        [48.0252, 71.7366],[33.5493, 92.3655],[62.7299, 92.2041] ]
    crop_size = (96, 112)
    src_pts = np.array(src_pts).reshape(5,2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model','-m', default='../my_face_model/model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
#net.cuda()
net.eval()
net.feature = True

class ImagePrepare(object):

    def __init__(self, path):
        super(ImagePrepare, self).__init__()

        self.path = path
        self.detector=MTCNN()
        self.imgs_name_list = []
        self.imgs_after_alignment = []
        self.get_alignment()

    def get_alignment(self):
        for i in range(len(self.landmarks)):
            landmark = self.landmarks[i]
            img = self.img_data[i]
            img_after_alignment = alignment(img, landmark)
            self.imgs_after_alignment.append(img_after_alignment)
        for i in range(len(self.imgs_after_alignment)):
            self.imgs_after_alignment[i] = self.imgs_after_alignment[i].transpose(2, 0, 1).reshape(1, 3, 112, 96)
            self.imgs_after_alignment[i] = (self.imgs_after_alignment[i] - 127.5)/ 128.0
        self.imgs_after_alignment = np.array(self.imgs_after_alignment)
        self.imgs_after_alignment = np.reshape(self.imgs_after_alignment, (len(self.imgs_after_alignment), 3, 112, 96))

    def cal_cosdistance(self, vec1, vec2):
        vec1 = np.reshape(vec1, (1, -1))
        vec2 = np.reshape(vec2, (-1, 1))
        length1 = np.sqrt(np.square(vec1).sum())
        length2 = np.sqrt(np.square(vec2).sum())
        cosdistance = vec1.dot(vec2) / (length1 * length2)
        cosdistance = cosdistance[0][0]
        return cosdistance

    def get_imgs_features(self, imgs_alignment):
        input_images = Variable(torch.from_numpy(imgs_alignment).float(), volatile=True)
        output_features = net(input_images)
        output_features = output_features.data.numpy()
        return output_features

'''
datas = ImagePrepare('./images')
input_images = datas.imgs_after_alignment
output_images_features = datas.get_imgs_features(input_images)
print(output_images_features.shape)
print(datas.imgs_name_list)
f1, f2 = output_images_features[0], output_images_features[-1]
cosdistance_test = datas.cal_cosdistance(f1, f2)
print(cosdistance_test)
det=datas.detector
'''
'''
cosdistance = np.reshape(f1, (1, -1)).dot(np.reshape(f2, (-1, 1)))/(np.sqrt(np.square(f1).sum())*np.sqrt(np.square(f2).sum())+1e-5)
print(cosdistance[0][0])
img_datas = ImagePrepare('../images')
print('=========', img_datas.img_name_list)

imgs_after_alignment = []
for i in range(len(img_datas.landmarks)):
    landmark = img_datas.landmarks[i]
    img = img_datas.img_data[i]
    img_after_alignment = alignment(img, landmark)
    imgs_after_alignment.append(img_after_alignment)

#print(len(imgs_after_alignment))

for i in range(len(imgs_after_alignment)):
    imgs_after_alignment[i] = imgs_after_alignment[i].transpose(2, 0, 1).reshape((1, 3, 112, 96))
    imgs_after_alignment[i] = (imgs_after_alignment[i] - 127.5) /128.0
#print(len(imgs_after_alignment))

imgs_after_alignment = np.array(imgs_after_alignment).reshape(len(imgs_after_alignment), 3, 112, 96)
#print(imgs_after_alignment.shape)

input_imgs = Variable(torch.from_numpy(imgs_after_alignment).float(), volatile=True)#.cuda()
output_features = net(input_imgs)

f = output_features.data.numpy()
f1, f2 = f[2], f[9]

cosdistance = np.reshape(f1, (1, -1)).dot(np.reshape(f2, (-1, 1)))/(np.sqrt(np.square(f1).sum())*np.sqrt(np.square(f2).sum())+1e-5)

print(cosdistance)
#print(output_features.data[0])
'''
