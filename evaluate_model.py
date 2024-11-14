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

#import model weights from config
load_dotenv("WEIGHTS_PATH.env")
unet_weights_path = os.getenv("msunet/MODEL_WEIGHTS/")
unet_classifier_weights_path = os.getenv("msunet_classifier/MODEL_WEIGHTS/")
vit_weights_path = os.getenv("vit/MODEL_WEIGHTS/")
mmca_weights_path = os.getenv("mmca/MODEL_WEIGHTS/")

def eval_msunet_classifier(FOLD_NUM: Optional[int] = None, checkpoint: Optional[Tuple[str, str]] = (unet_weights_path, unet_classifier_weights_path)):
    assert isinstance(checkpoint, Tuple) 
    pass

def eval_vit_classifier(FOLD_NUM: Optional[int] = None, checkpoint: Optional[str] = vit_weights_path):
    pass

def eval_jointfusion_classifier(FOLD_NUM: Optional[int] = None, checkpoint: Optional[Tuple[str, str, str]] = (unet_weights_path, unet_classifier_weights_path, vit_weights_path)):
    assert isinstance(checkpoint, Tuple)
    pass

def eval_mmca(FOLD_NUM: Optional[int] = None, checkpoint: Optional[str]: mmca_weights_path):
    pass
