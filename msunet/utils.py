#stl
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union
import math
import os
import warnings

#data handling
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image

#stats
import scipy
import sklearn

#network
import networkx as nx

#vis
import matplotlib.pyplot as plt
import seaborn as sns

#torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    f"""
    ARGS:
        TARGET_FOLD (int): The fold number to split the data into
        h5_data (str): Path to the h5 file containing the image and mask data
        labels_df (pd.DataFrame): DataFrame containing the labels and fold numbers
        PHASE (str): The phase of the data ("train", "test")
        target_size (int): Size to resize the images to
    """
    def __init__(self, TARGET_FOLD, h5_data, labels_df, phase, target_size = (512,512)):

        #0. for forward pass
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.targetSize = target_size

        #1. empty init data
        self.images = []
        self.masks = []
        
        #2. split data accordingly
        self.phase = phase
        self.train_test_split(TARGET_FOLD, h5_data, labels_df)
    

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = processImage(self.images[idx], self.targetSize)
        mask = processImage(self.masks[idx], self.targetSize)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
        
    def train_test_split(self, TARGET_FOLD, h5_data, labels_df): 
        if self.phase in ["train", "TRAIN", "Train"]:
            target_index = labels_df[labels_df["foldNum"] != TARGET_FOLD].index.to_list()

        elif self.phase in ["test", "TEST", "Test"]:
            target_index = labels_df[labels_df["foldNum"] == TARGET_FOLD].index.to_list()
        
        for i in tqdm(range(len(h5_data["image"])), desc = "Splitting FOLD %s for %s" % (TARGET_FOLD, self.phase)):
            if i in target_index:
                self.images.append(h5_data["image"][i])
            else:
                self.masks.append(h5_data["mask"][i])

        self.images = np.array(self.images)
        self.masks = np.array(self.masks)
        

def processImage(image: np.array, targetSize: tuple = (512,512)):
    r"""
        Processes a grayscale image stored as a NumPy .npy file, performs scaling, type conversion, 
        resizing, and cropping/padding to return a square PIL image of specified target size.

        Parameters:
        - image (np.array): the image
        - targetSize (tuple): The desired size of the final image. If None, the final image will be 512x512.

        Returns:
        - PIL.Image: The processed PIL Image.
    """ 
    if(npImg.dtype != np.uint8):
        # min max scale
        if (np.max(npImg) - np.min(npImg) != 0): 
            npImg = (255 * (npImg - np.min(npImg)) / (np.max(npImg) - np.min(npImg))).astype(np.uint8)
        else: 
            npImg = (255*npImg).astype(np.uint8)

    
    pilImg = Image.fromarray(npImg, 'L')

    #scale after resize
    scale = targetSize[0]/max(pilImg.width, pilImg.height)
    
    if pilImg.width > pilImg.height: 
        newSize = (targetSize[0], int(pilImg.height * scale))
        
    elif pilImg.width < pilImg.height: 
        newSize = (int(pilImg.width * scale), targetSize[1])
        
    else: 
        newSize = targetSize

    #lanczos scaling
    pilImg = pilImg.resize(newSize, Image.LANCZOS)
    finalImg = Image.new('L', targetSize)
    
    pastePosition = ((targetSize[0]-newSize[0])//2, (targetSize[1]-newSize[1])//2)
    finalImg.paste(pilImg, pastePosition)
    
    return finalImg