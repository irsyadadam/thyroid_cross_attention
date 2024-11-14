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

if __name__ == "__main__":
    
    #load configs
    load_dotenv("WEIGHTS_PATH.env")

    #CLI args
    parser = argparse.ArgumentParser(description="Evaluation of the Multi-Modal Cross Attention of Thyroid Cine Ultrasound Encodings. Choose backbone with Flag Params, MMCA decoder will use specified backbones if chosen.")

    parser.add_argument('--MSUNET', action = "store_true", help = '(Bool) Evaluate the MSUNet + Classifier for [FOLD].')
    parser.add_argument('--VIT', action = "store_true", help = "(Bool) Evaluate the ViT + Classifier for [FOLD].")
    parser.add_argument('--JOINTFUSION', action = "store_true", help = "(Bool) Evaluate the MSUNet + ViT + Classifier for [FOLD].")
    parser.add_argument('--MMCA', type = int, default = None, help = "(Bool) Evaluate the MMCA for [FOLD], specify backbone with flag params")
    parser.add_argument('--FOLD', type = int, default = None, help = "(Int) Fold num to do task on.")

    args = parser.parse_args()
    if args.FOLD is None:
        print("Fold num not provided. Exiting.")
        exit(1)

    if args.MS_UNet:
        eval_msunet_classifier(FOLD_NUM = args.FOLD)
        if args.MMCA:
            eval_mmca(backbone = args.MS_UNet)

    elif args.VIT:
        eval_vit_classifier(FOLD_NUM = args.FOLD)
        if args.MMCA:
            eval_mmca(backbone = args.MS_UNet)

    elif args.JOINTFUSION:
        eval_jointfusion_classifier(FOLD_NUM = args.FOLD)
        if args.MMCA:
            eval_mmca(backbone = args.MS_UNet)



