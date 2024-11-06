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
import warnings
import argparse
from dotenv import load_dotenv


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

#TODO: MAKE SURE THAT TRAINING DOES NOT OVERWRITE EXISTING WEIGHS IF ALR TRAINED UNLESS WITH FORCE FLAG

from train import train_msunet

if __name__ == "__main__":

    #load configs
    load_dotenv("UNET_CONFIGS.env")

    initial_checkpoint = os.getenv("UCLA_WEIGHTS")
    output_weights_folder = os.getenv("OUTPUT_WEIGHTS")

    #CLI args
    parser = argparse.ArgumentParser(description="Fine-tunes the MSUNet (Radhachandran et al. 2023) starting from the UCLA Image checkpoint, and changes weights towards seg of stanford thyroid cine.")

    parser.add_argument('--TRAIN', action = "store_true", help = 'Fine tune MSUNet on Stanford Dataset, using specified fold flag')
    parser.add_argument('--FORCE_TRAIN', action = "store_true", help = "Flag to overwrite existing weights for specified fold.")
    parser.add_argument('--FOLD', type = int, default = None, help = "Fold num to do task on.")

    args = parser.parse_args()
    if args.FOLD is None:
        print("Fold num not provided. Exiting.")
        exit(1)

    if args.TRAIN or args.FORCE_TRAIN:
        #check if existing weights:
        if not os.path.exists(output_weights_folder + "FOLD_%s/" % args.FOLD) and args.FORCE_TRAIN == False:
            train_msunet(FOLD_NUM = args.FOLD)
        elif os.path.exists(output_weights_folder + "FOLD_%s/" % args.FOLD) and args.FORCE_TRAIN == False:
            print(f"Weights for fold {args.FOLD} already exist. Use --FORCE_TRAIN to overwrite.")
            exit(1)
        else:
            train_msunet(FOLD_NUM = args.FOLD)

