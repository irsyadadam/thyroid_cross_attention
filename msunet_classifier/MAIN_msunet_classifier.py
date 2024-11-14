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

from MAIN_msunet_classifier import train_msunet_classifier

if __name__ == "__main__":
    
    #load configs
    load_dotenv("WEIGHTS_PATH.env")

    #CLI args
    parser = argparse.ArgumentParser(description="Training a classifier to encode the output images from msunet")
    parser.add_argument('--FOLD', type = int, default = None, help = "(Int) Fold num to do task on.")

    args = parser.parse_args()
    if args.FOLD is None:
        print("Fold num not provided. Exiting.")
        exit(1)
    
    train_msunet_classifier(FOLD_NUM = args.FOLD)