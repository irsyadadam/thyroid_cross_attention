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

        #get targets, only retrieve when getting __getitem__ is called
        self.phase = phase
        self.target_index = self.get_targets(TARGET_FOLD, h5_data, labels_df)


        #no copy, only pointer to dataset
        self.h5_data = h5_data


    def __len__(self):
        return len(self.target_index)

    def __getitem__(self, idx):
        target = self.target_index[idx]
        image = processImage(self.h5_data["image"][target], self.targetSize)
        mask = processImage(self.h5_data["mask"][target], self.targetSize)

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
        
    def get_targets(self, TARGET_FOLD, h5_data, labels_df): 
        if self.phase in ["train", "TRAIN", "Train"]:
            target_index = labels_df[labels_df["foldNum"] != TARGET_FOLD].index.to_list()

        elif self.phase in ["test", "TEST", "Test"]:
            target_index = labels_df[labels_df["foldNum"] == TARGET_FOLD].index.to_list()
        
        return target_index

def processImage(npImg: np.array, targetSize: tuple = (512,512)):
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