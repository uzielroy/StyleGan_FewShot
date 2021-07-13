"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""
import glob
import zipfile
import os
import argparse
import itertools
import io
import os
import argparse
import pickle
import math
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import grad
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, Miner
from train import accumulate, sample_data, adjust_lr
from metric.inception import InceptionV3
from metric.metric import get_fake_images_and_acts, compute_fid
from loss.AdaBIGGANLoss import AdaBIGGANLoss
from DiffAugment_pytorch import DiffAugment
from torch.utils.data import Dataset, DataLoader

import tensorflow as tf
def calc_distances(p0, points):
    return ((p0 - points)**2).sum(axis=1)

def graipher(pts, K):
    idx_pts = np.zeros(K)
    farthest_pts = np.zeros((K, 2))
    rand_idx = np.random.randint(len(pts))
    farthest_pts[0] = pts[rand_idx]
    idx_pts[0]=rand_idx
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, K):
        max_idx = np.argmax(distances)
        farthest_pts[i] = pts[max_idx]
        idx_pts[i] = max_idx
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts, idx_pts.astype(int)




def _parse_image_function(example_proto):
 train_feature_description = {'image': tf.io.FixedLenFeature([], tf.string),
 }
 return tf.io.parse_single_example(example_proto, train_feature_description)

def images_pca(images,k=30):
    resize = transforms.Resize(256)
    data = np.zeros((len(images),256*256))
    idx = 0
    for img in images:
        data[idx] = np.array(resize(Image.open(io.BytesIO(img))).convert('L')).reshape(1,-1)
        idx = idx+1
    pca = PCA(2)
    converted_data = pca.fit_transform(data)
    pts,indices = graipher(converted_data,k)
    print(indices)
    reduced_images = list(np.array(images)[indices])
    save_reduced_images(reduced_images)
    print(len(reduced_images))
    return reduced_images

def save_reduced_images(images):
    os.makedirs('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/monet_reduced', exist_ok=True)
    i = 0
    for img in images:
        Image.open(io.BytesIO(img)).convert('RGB').save('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/monet_reduced/'+str(i).zfill(3)+'.png')
        i = i+1
def make_dataset_kaggle(path,monet=True):
    train_feature_description = {
    'image': tf.io.FixedLenFeature([], tf.string),
    }
    extension = "tfrec.zip"
    if monet:
        for item in os.listdir('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/'): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('monet'): # check for "148.tfrec.zip" extension
                file_name = file_name = os.path.join(path,item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/monet') # extract file to dir
                zip_ref.close() # close file
        for item in os.listdir(os.path.join(path,'monet')):
            if item.endswith('tfrec.zip') and item.startswith('monet'):
                file_name = os.path.join(path,item)
                zip_ref = zipfile.ZipFile(file_name)
                zip_ref.extractall(os.path.join(path,'monet'))
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob(os.path.join(path,'monet/*.tfrec'))
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          print("555555555555555555555555", i)
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images
        print("HEREEE:", len(train_images))
        train_images = images_pca(train_images)
        print("THEREEE:", len(train_images))

    else:
        for item in os.listdir('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/'): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('photo'): # check for "148.tfrec.zip" extension
                file_name = file_name = os.path.join(path,item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/photo') # extract file to dir
                zip_ref.close() # close file
        for item in os.listdir(os.path.join(path,'photo')):
            if item.endswith('tfrec.zip') and item.startswith('photo'):
                file_name = os.path.join(path,item)
                zip_ref = zipfile.ZipFile(file_name)
                zip_ref.extractall(os.path.join(path,'photo'))
                zip_ref.close() # close file
        for item in os.listdir(os.path.join(path,'photo')): # loop through items in dir
            if item.endswith('tfrec.zip') and item.startswith('photo'): # check for "148.tfrec.zip" extension
                file_name = os.path.abspath(item) # get full path of files
                zip_ref = zipfile.ZipFile(file_name) # create zipfile object
                zip_ref.extractall(os.path.join(path,'photo/*.tfrec')) # extract file to dir
                zip_ref.close() # close file

        train_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
        }
        train_files = glob.glob(os.path.join(path,'photo/*.tfrec'))
        train_ids = []
        train_class = []
        train_images = []
        for i in train_files:
          train_image_dataset = tf.data.TFRecordDataset(i)
          train_image_dataset = train_image_dataset.map(_parse_image_function)
          images = [image_features['image'].numpy() for image_features in train_image_dataset]
          train_images = train_images + images

    return train_images


make_dataset_kaggle('/content/Pnina/MyDrive/StyleGan_FewShot/kaggle_dataset/',True)
