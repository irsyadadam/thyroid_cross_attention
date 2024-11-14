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
import argparse

#data handling
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

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
from torch.utils.data import DataLoader, random_split, Dataset


#IMPORT
from models import MSU_Net
from utils import SegmentationDataset
from train_utils import training, CombinedLoss
from dotenv import load_dotenv
import h5py

import warnings
warnings.filterwarnings("ignore")

#load env var
load_dotenv("../DATA_PATH.env")
h5_path = os.getenv("h5_PATH")
labels_df_path = os.getenv("labeldf_PATH")

#load data
labels_df = pd.read_csv(labels_df_path, index_col = [0])
data = h5py.File(h5_path)

#load configs
load_dotenv("UNET_CONFIGS.env")

initial_checkpoint = os.getenv("UCLA_WEIGHTS")
output_weights_folder = os.getenv("OUTPUT_WEIGHTS")

BATCH_SIZE = int(os.getenv("UNET_BATCH_SIZE"))
LR = float(os.getenv("UNET_LR"))
EPOCHS = int(os.getenv("UNET_EPOCHS"))


def train_msunet(FOLD_NUM: Optional[int], checkpoint: Optional[str] = initial_checkpoint):
    r"""
    Training Step on Dataset, starting from checkpoint

    ARGS:
        FOLD_NUM: int is the fold number specified. Only trains on the fold num, not eval
    """
    print("-----------Fine Tune MSUnet from Initial Checkpoint-----------\n")
    #load data
    print("Loading fold %s dataset:" % FOLD_NUM,  end = " ")
    train_data = SegmentationDataset(TARGET_FOLD = FOLD_NUM, h5_data = data, phase = "Train", labels_df = labels_df)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    test_data = SegmentationDataset(TARGET_FOLD = FOLD_NUM, h5_data = data, phase = "Test", labels_df = labels_df)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    print("success. \n")

    print("Loading initial checkpoint: ",  end = " ")
    #load model etc
    MODEL_WEIGHTS = checkpoint
    device_ids = [0]  # replace with your GPU ids if you have multiple GPUs
    DEVICE = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(MSU_Net(), device_ids = device_ids).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location = DEVICE))
    print(f"{checkpoint}. success. \n")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.9))
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.2, focal_weight=0.1, pos_weight=torch.tensor([10]).to(DEVICE), gamma=4) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.1*LR, verbose=True)

    print("-----------Train Step-----------\n")
    trainLoss, testLoss, trainDICE, testDICE = training(model, train_loader, optimizer, criterion, scheduler, test_loader, DEVICE, FOLD_NUM = FOLD_NUM, epochs = EPOCHS, model_file_ext = "MODEL_WEIGHTS/FOLD_%s/unet_tuned" % FOLD_NUM)
    
    print("-----------Complete-----------\n")