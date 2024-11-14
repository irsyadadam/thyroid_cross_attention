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

#load configs
load_dotenv("WEIGHTS_PATH.env")
unet_weights_path = os.getenv("unet_weights_path")

def train_msunet_classifier(FOLD_NUM: Optional[int], checkpoint_dir: Optional[str] = unet_weights_path):

    print("----------Training Classifier on MSUNet Output----------")
    
    #forward pass through msunet with specified fold
    

