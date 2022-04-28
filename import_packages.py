import os
import sys
import logging
import time
import random
import warnings
sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')

import audioread
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager # Q: how does this work?
from pathlib import Path
from typing import List
from typing import Optional
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, GroupKFold

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

import albumentations as A
import albumentations.pytorch.transforms as T
