import os
import json
import numpy as np
import torch as pt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

from src.logger import Logger
from src.scoring import bc_scoring, bc_score_names, nanmean
from config import config_data, config_model, config_runtime
from data_handler import Dataset, collate_batch_data
from src.dataset import select_by_sid, select_by_max_ba, select_by_interface_types
from model import Model
from src.data_encoding import categ_to_resnames

t60_labels = {'11.247': '4lvx', '13.768': '1arj', '8.105': '2g5k', '26.912': '1yls', '42.142': '1nbk', '18.214': '4nyb', '-27.978': '3q3z', '209.580': '5j02', '27.511': '4qjh', '48.215': '1yrj', '8.379': '1ei2', '-11.394': '2lwk', '-0.796': '2l8h', '-15.046': '1lvj', '19.074': '4qlm', '11.685': '1raw', '-8.21': '2quw', '64.872': '3gx3', '-10.376': '5vci', '24.473': '4k31', '20.365': '1ntb', '4.698': '1eht', '87.881': '1y90', '58.355': '1hr2', '25.734': '4pcj', '26.33': '3mei', '-43.498': '5dh8', '-5.382': '4rge', '3.849': '2fcy', '-69.036': '4meg', '-14.864': '2nok', '30.507': '3q50', '35.9': '3skt', '-100.220': '2kgp', '81.222': '5d5l', '-14.325': '1uts', '6.654': '1fyp', '0.909': '1aju', '-24.478': '3vrs', '5.564': '1byj', '7.386': '4xw7', '22.586': '6fz0', '-2.696': '3tzr', '38.26': '3c44', '-9.159': '1uud', '5.076': '1tob', '-19.661': '3bnq', '127.495': '1f1t', '-1.176': '3oxe', '11.726': '5u3g', '15.428': '4mgm', '-15.901': '2mxs', '2.682': '2ktz', '-7.41': '5fj1', '-10.124': '1qd3', '3.759': '4frg', '3.07': '2yie', '-26.502': '3fu2', '2.626': '6hag', '15.068': '5o69'}
t60_mapping = {'4meg': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0], '4rge': [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], '4pcj': [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], '1byj': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0], '3tzr': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], '2yie': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4mgm': [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], '4qjh': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '3q50': [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0], '5o69': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], '3mei': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1], '5d5l': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], '2l8h': [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], '3skt': [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], '1ei2': [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], '2fcy': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], '1ntb': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], '1y90': [0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], '1aju': [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], '1yls': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1], '1uts': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], '1raw': [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], '3gx3': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], '1f1t': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0], '5u3g': [0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], '2quw': [1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], '1nbk': [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], '2kgp': [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], '5j02': [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], '5fj1': [0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], '1hr2': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0], '3bnq': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], '4frg': [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '2mxs': [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], '4lvx': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '4xw7': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], '3c44': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], '1eht': [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0], '3oxe': [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], '3q3z': [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], '1arj': [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], '1uud': [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0], '2lwk': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], '4nyb': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0], '4qlm': [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '5dh8': [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0], '4k31': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0], '1yrj': [0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], '3fu2': [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1], '1qd3': [0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0], '2g5k': [0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0], '2ktz': [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0], '1fyp': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], '6fz0': [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0], '5vci': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], '1lvj': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], '3vrs': [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '6hag': [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0], '1tob': [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], '2nok': [0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
def setup_dataloader(config_data, sids_selection_filepath):
    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))

    # create dataset
    dataset = Dataset(config_data['dataset_filepath'])

    # data selection criteria
    m = select_by_sid(dataset, sids_sel) # select by sids
    m &= select_by_max_ba(dataset, config_data['max_ba'])  # select by max assembly count
    # m &= (dataset.sizes[:,0] <= config_data['max_size']) # select by max size
    # m &= (dataset.sizes[:,1] >= config_data['min_num_res'])  # select by min size
    m &= select_by_interface_types(dataset, categ_to_resnames['rna'], np.concatenate(config_data['r_types']))  # select by interface type

    # update dataset selection
    dataset.update_mask(m)

    # set dataset types for labels
    dataset.set_types(categ_to_resnames['rna'], config_data['r_types'])

    # define data loader
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=config_runtime['batch_size'], shuffle=True, num_workers=8, collate_fn=collate_batch_data, pin_memory=True, prefetch_factor=2)

    return dataloader


def eval_step(model, device, batch_data, criterion, pos_ratios, pos_weight_factor, global_step):
    # unpack data
    X, ids_topk, q, M, y = [data.to(device) for data in batch_data]
    yx = t60_mapping[t60_labels[str(round(X[0][0].item(), 3))]]
    y = pt.zeros((len(yx), 5))
    for i, value in enumerate(yx):
        y[i, 3] = float(value)
    y = y.to(device)
    # run model
    z = model.forward(X, ids_topk, q, M)

    # compute weighted loss
    pos_ratios += (pt.mean(y,dim=0).detach() - pos_ratios) / (1.0 + np.sqrt(global_step))
    criterion.pos_weight = pos_weight_factor * (1.0 - pos_ratios) / (pos_ratios + 1e-6)
    dloss = criterion(z, y)

    # re-weighted losses
    loss_factors = (pos_ratios / pt.sum(pos_ratios)).reshape(1,-1)
    losses = (loss_factors * dloss) / dloss.shape[0]

    return losses, y.detach(), pt.sigmoid(z).detach()


def scoring(eval_results, device=pt.device('cpu')):
    # compute sum losses and scores for each entry
    sum_losses, scores = [], []
    for losses, y, p in eval_results:
        sum_losses.append(pt.sum(losses, dim=0))
        scores.append(bc_scoring(y, p))

    # average scores
    m_losses = pt.mean(pt.stack(sum_losses, dim=0), dim=0).numpy()
    m_scores = nanmean(pt.stack(scores, dim=0)).numpy()

    # pack scores
    scores = {'loss': float(np.sum(m_losses))}
    for i in range(m_losses.shape[0]):
        scores[f'{i}/loss'] = m_losses[i]
        for j in range(m_scores.shape[0]):
            scores[f'{i}/{bc_score_names[j]}'] = m_scores[j,i]

    return scores


def logging(logger, writer, scores, global_step, pos_ratios, step_type):
    # debug print
    pr_str = ', '.join([f"{r:.4f}" for r in pos_ratios])
    logger.print(f"{step_type}> [{global_step}] loss={scores['loss']:.4f}, pos_ratios=[{pr_str}]")

    # store statistics
    summary_stats = {k:scores[k] for k in scores if not np.isnan(scores[k])}
    summary_stats['global_step'] = int(global_step)
    summary_stats['pos_ratios'] = list(pos_ratios.cpu().numpy())
    summary_stats['step_type'] = step_type
    logger.store(**summary_stats)

    # detailed information
    # for key in scores:
    #     writer.add_scalar(step_type+'/'+key, scores[key], global_step)

    # debug print
    for c in np.unique([key.split('/')[0] for key in scores if len(key.split('/')) == 2]):
        logger.print(f'[{c}] loss={scores[c+"/loss"]:.3f}, ' + ', '.join([f'{sn}={scores[c+"/"+sn]:.3f}' for sn in bc_score_names]))


def train(config_data, config_model, config_runtime, output_path):
    # create logger
    logger = Logger(output_path, 'train')

    # print configuration
    logger.print(">>> Configuration")
    logger.print(config_data)
    logger.print(config_runtime)

    # define device
    device = pt.device(config_runtime['device'])

    # create model
    model = Model(config_model)
    # debug print
    logger.print(">>> Model")
    logger.print(model)
    logger.print(f"> {sum([int(pt.prod(pt.tensor(p.shape))) for p in model.parameters()])} parameters")

    # reload model if configured
    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath))
        # get last global step
        global_step = json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(logger.log_lst_filepath, 'r')][-1])['pos_ratios'])).float().to(device)
    else:
        # starting global step
        global_step = 0
        # dynamic positive weight
        pos_ratios = 0.5*pt.ones(len(config_data['r_types']), dtype=pt.float).to(device)

    # debug print
    logger.print(">>> Loading data")

    # setup dataloaders
    dataloader_train = setup_dataloader(config_data, config_data['train_selection_filepath'])
    dataloader_test = setup_dataloader(config_data, config_data['test_selection_filepath'])

    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")
    logger.print(f"> testing data size: {len(dataloader_test)}")

    # debug print
    logger.print(">>> Starting training")

    # send model to device
    model = model.to(device)

    # define losses functions
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])

    # restart timer
    logger.restart_timer()

    # summary writer
    # writer = SummaryWriter(os.path.join(output_path, 'tb'))
    writer = SummaryWriter("/Users/riddhishthakare/Downloads/PeSTo/runs/50_3_4to6")


    # min loss initial value
    min_loss = 1e9

    # quick training step on largest data: memory check and pre-allocation
    batch_data = collate_batch_data([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    losses, _, _ = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
    loss = pt.sum(losses)
    loss.backward()
    optimizer.step()

    # start training
    for epoch in range(config_runtime['num_epochs']):
        # train mode
        print("Epoch no --", epoch)
        model = model.train()

        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):
            # global step
            global_step += 1

            # set gradient to zero
            optimizer.zero_grad()

            # forward propagation
            losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

            # backward propagation
            loss = pt.sum(losses)
            loss.backward()

            # optimization step
            optimizer.step()

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

            writer.add_scalar("loss x epoch", loss, epoch)
            with pt.no_grad():
                # scores evaluation results and reset buffer
                scores = scoring(train_results, device=device)
                train_results = []
                logging(logger, writer, scores, global_step, pos_ratios, "train")

            # log step
            if (global_step+1) % config_runtime["log_step"] == 0:
                # process evaluation results
                with pt.no_grad():
                    # scores evaluation results and reset buffer
                    scores = scoring(train_results, device=device)
                    train_results = []

                    # logging
                    # logging(logger, writer, scores, global_step, pos_ratios, "train")


                    # save model checkpoint
                    model_filepath = os.path.join(output_path, 'model_ckpt.pt')
                    # pt.save(model.state_dict(), model_filepath)

            # evaluation step
            if (global_step+1) % config_runtime["eval_step"] == 0:
                # evaluation mode
                model = model.eval()

                with pt.no_grad():
                    # evaluate model
                    test_results = []
                    for step_te, batch_test_data in enumerate(dataloader_test):
                        # forward propagation
                        losses, y, p = eval_step(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

                        # store evaluation results
                        test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                        # stop evaluating
                        if step_te >= config_runtime['eval_size']:
                            break

                    # scores evaluation results
                    scores = scoring(test_results, device=device)

                    # logging
                    logging(logger, writer, scores, global_step, pos_ratios, "test")

                    # save model and update min loss
                    if min_loss >= scores['loss']:
                        # update min loss
                        min_loss = scores['loss']
                        # save model
                        model_filepath = os.path.join(output_path, 'model.pt')
                        logger.print("> saving model at {}".format(model_filepath))
                        # pt.save(model.state_dict(), model_filepath)

                # back in train mode
                model = model.train()
    
    writer.close()

def train1(config_data, config_model, config_runtime, output_path):
    # create logger
    logger = Logger(output_path, 'train')
    writer = SummaryWriter("/blue/yanjun.li/riddhishthakare/PeSTo/runs/40_SMOL_1.4_semi_8A_real")

    device = pt.device(config_runtime['device'])
    model = Model(config_model)
    value=False

    unfreeze = ['sum.25.su.nqm.0.weight', 'sum.25.su.nqm.0.bias', 'sum.25.su.nqm.2.weight', 'sum.25.su.nqm.2.bias', 'sum.25.su.nqm.4.weight', 'sum.25.su.nqm.4.bias', 'sum.25.su.eqkm.0.weight', 'sum.25.su.eqkm.0.bias', 'sum.25.su.eqkm.2.weight', 'sum.25.su.eqkm.2.bias', 'sum.25.su.eqkm.4.weight', 'sum.25.su.eqkm.4.bias', 'sum.25.su.epkm.0.weight', 'sum.25.su.epkm.0.bias', 'sum.25.su.epkm.2.weight', 'sum.25.su.epkm.2.bias', 'sum.25.su.epkm.4.weight', 'sum.25.su.epkm.4.bias', 'sum.25.su.evm.0.weight', 'sum.25.su.evm.0.bias', 'sum.25.su.evm.2.weight', 'sum.25.su.evm.2.bias', 'sum.25.su.evm.4.weight', 'sum.25.su.evm.4.bias', 'sum.25.su.qpm.0.weight', 'sum.25.su.qpm.0.bias', 'sum.25.su.qpm.2.weight', 'sum.25.su.qpm.2.bias', 'sum.25.su.qpm.4.weight', 'sum.25.su.qpm.4.bias', 'sum.25.su.ppm.0.weight', 'sum.24.su.nqm.0.weight', 'sum.24.su.nqm.0.bias', 'sum.24.su.nqm.2.weight', 'sum.24.su.nqm.2.bias', 'sum.24.su.nqm.4.weight', 'sum.24.su.nqm.4.bias', 'sum.24.su.eqkm.0.weight', 'sum.24.su.eqkm.0.bias', 'sum.24.su.eqkm.2.weight', 'sum.24.su.eqkm.2.bias', 'sum.24.su.eqkm.4.weight', 'sum.24.su.eqkm.4.bias', 'sum.24.su.epkm.0.weight', 'sum.24.su.epkm.0.bias', 'sum.24.su.epkm.2.weight', 'sum.24.su.epkm.2.bias', 'sum.24.su.epkm.4.weight', 'sum.24.su.epkm.4.bias', 'sum.24.su.evm.0.weight', 'sum.24.su.evm.0.bias', 'sum.24.su.evm.2.weight', 'sum.24.su.evm.2.bias', 'sum.24.su.evm.4.weight', 'sum.24.su.evm.4.bias', 'sum.24.su.qpm.0.weight', 'sum.24.su.qpm.0.bias', 'sum.24.su.qpm.2.weight', 'sum.24.su.qpm.2.bias', 'sum.24.su.qpm.4.weight', 'sum.24.su.qpm.4.bias', 'sum.24.su.ppm.0.weight', 'sum.23.su.nqm.0.weight', 'sum.23.su.nqm.0.bias', 'sum.23.su.nqm.2.weight', 'sum.23.su.nqm.2.bias', 'sum.23.su.nqm.4.weight', 'sum.23.su.nqm.4.bias', 'sum.23.su.eqkm.0.weight', 'sum.23.su.eqkm.0.bias', 'sum.23.su.eqkm.2.weight', 'sum.23.su.eqkm.2.bias', 'sum.23.su.eqkm.4.weight', 'sum.23.su.eqkm.4.bias', 'sum.23.su.epkm.0.weight', 'sum.23.su.epkm.0.bias', 'sum.23.su.epkm.2.weight', 'sum.23.su.epkm.2.bias', 'sum.23.su.epkm.4.weight', 'sum.23.su.epkm.4.bias', 'sum.23.su.evm.0.weight', 'sum.23.su.evm.0.bias', 'sum.23.su.evm.2.weight', 'sum.23.su.evm.2.bias', 'sum.23.su.evm.4.weight', 'sum.23.su.evm.4.bias', 'sum.23.su.qpm.0.weight', 'sum.23.su.qpm.0.bias', 'sum.23.su.qpm.2.weight', 'sum.23.su.qpm.2.bias', 'sum.23.su.qpm.4.weight', 'sum.23.su.qpm.4.bias', 'sum.23.su.ppm.0.weight', 'sum.22.su.nqm.0.weight', 'sum.22.su.nqm.0.bias', 'sum.22.su.nqm.2.weight', 'sum.22.su.nqm.2.bias', 'sum.22.su.nqm.4.weight', 'sum.22.su.nqm.4.bias', 'sum.22.su.eqkm.0.weight', 'sum.22.su.eqkm.0.bias', 'sum.22.su.eqkm.2.weight', 'sum.22.su.eqkm.2.bias', 'sum.22.su.eqkm.4.weight', 'sum.22.su.eqkm.4.bias', 'sum.22.su.epkm.0.weight', 'sum.22.su.epkm.0.bias', 'sum.22.su.epkm.2.weight', 'sum.22.su.epkm.2.bias', 'sum.22.su.epkm.4.weight', 'sum.22.su.epkm.4.bias', 'sum.22.su.evm.0.weight', 'sum.22.su.evm.0.bias', 'sum.22.su.evm.2.weight', 'sum.22.su.evm.2.bias', 'sum.22.su.evm.4.weight', 'sum.22.su.evm.4.bias', 'sum.22.su.qpm.0.weight', 'sum.22.su.qpm.0.bias', 'sum.22.su.qpm.2.weight', 'sum.22.su.qpm.2.bias', 'sum.22.su.qpm.4.weight', 'sum.22.su.qpm.4.bias', 'sum.22.su.ppm.0.weight', 'sum.21.su.nqm.0.weight', 'sum.21.su.nqm.0.bias', 'sum.21.su.nqm.2.weight', 'sum.21.su.nqm.2.bias', 'sum.21.su.nqm.4.weight', 'sum.21.su.nqm.4.bias', 'sum.21.su.eqkm.0.weight', 'sum.21.su.eqkm.0.bias', 'sum.21.su.eqkm.2.weight', 'sum.21.su.eqkm.2.bias', 'sum.21.su.eqkm.4.weight', 'sum.21.su.eqkm.4.bias', 'sum.21.su.epkm.0.weight', 'sum.21.su.epkm.0.bias', 'sum.21.su.epkm.2.weight', 'sum.21.su.epkm.2.bias', 'sum.21.su.epkm.4.weight', 'sum.21.su.epkm.4.bias', 'sum.21.su.evm.0.weight', 'sum.21.su.evm.0.bias', 'sum.21.su.evm.2.weight', 'sum.21.su.evm.2.bias', 'sum.21.su.evm.4.weight', 'sum.21.su.evm.4.bias', 'sum.21.su.qpm.0.weight', 'sum.21.su.qpm.0.bias', 'sum.21.su.qpm.2.weight', 'sum.21.su.qpm.2.bias', 'sum.21.su.qpm.4.weight', 'sum.21.su.qpm.4.bias', 'sum.21.su.ppm.0.weight', 'sum.20.su.nqm.0.weight', 'sum.20.su.nqm.0.bias', 'sum.20.su.nqm.2.weight', 'sum.20.su.nqm.2.bias', 'sum.20.su.nqm.4.weight', 'sum.20.su.nqm.4.bias', 'sum.20.su.eqkm.0.weight', 'sum.20.su.eqkm.0.bias', 'sum.20.su.eqkm.2.weight', 'sum.20.su.eqkm.2.bias', 'sum.20.su.eqkm.4.weight', 'sum.20.su.eqkm.4.bias', 'sum.20.su.epkm.0.weight', 'sum.20.su.epkm.0.bias', 'sum.20.su.epkm.2.weight', 'sum.20.su.epkm.2.bias', 'sum.20.su.epkm.4.weight', 'sum.20.su.epkm.4.bias', 'sum.20.su.evm.0.weight', 'sum.20.su.evm.0.bias', 'sum.20.su.evm.2.weight', 'sum.20.su.evm.2.bias', 'sum.20.su.evm.4.weight', 'sum.20.su.evm.4.bias', 'sum.20.su.qpm.0.weight', 'sum.20.su.qpm.0.bias', 'sum.20.su.qpm.2.weight', 'sum.20.su.qpm.2.bias', 'sum.20.su.qpm.4.weight', 'sum.20.su.qpm.4.bias', 'sum.20.su.ppm.0.weight','sum.19.su.nqm.0.weight', 'sum.19.su.nqm.0.bias', 'sum.19.su.nqm.2.weight', 'sum.19.su.nqm.2.bias', 'sum.19.su.nqm.4.weight', 'sum.19.su.nqm.4.bias', 'sum.19.su.eqkm.0.weight', 'sum.19.su.eqkm.0.bias', 'sum.19.su.eqkm.2.weight', 'sum.19.su.eqkm.2.bias', 'sum.19.su.eqkm.4.weight', 'sum.19.su.eqkm.4.bias', 'sum.19.su.epkm.0.weight', 'sum.19.su.epkm.0.bias', 'sum.19.su.epkm.2.weight', 'sum.19.su.epkm.2.bias', 'sum.19.su.epkm.4.weight', 'sum.19.su.epkm.4.bias', 'sum.19.su.evm.0.weight', 'sum.19.su.evm.0.bias', 'sum.19.su.evm.2.weight', 'sum.19.su.evm.2.bias', 'sum.19.su.evm.4.weight', 'sum.19.su.evm.4.bias', 'sum.19.su.qpm.0.weight', 'sum.19.su.qpm.0.bias', 'sum.19.su.qpm.2.weight', 'sum.19.su.qpm.2.bias', 'sum.19.su.qpm.4.weight', 'sum.19.su.qpm.4.bias', 'sum.19.su.ppm.0.weight','sum.18.su.nqm.0.weight', 'sum.18.su.nqm.0.bias', 'sum.18.su.nqm.2.weight', 'sum.18.su.nqm.2.bias', 'sum.18.su.nqm.4.weight', 'sum.18.su.nqm.4.bias', 'sum.18.su.eqkm.0.weight', 'sum.18.su.eqkm.0.bias', 'sum.18.su.eqkm.2.weight', 'sum.18.su.eqkm.2.bias', 'sum.18.su.eqkm.4.weight', 'sum.18.su.eqkm.4.bias', 'sum.18.su.epkm.0.weight', 'sum.18.su.epkm.0.bias', 'sum.18.su.epkm.2.weight', 'sum.18.su.epkm.2.bias', 'sum.18.su.epkm.4.weight', 'sum.18.su.epkm.4.bias', 'sum.18.su.evm.0.weight', 'sum.18.su.evm.0.bias', 'sum.18.su.evm.2.weight', 'sum.18.su.evm.2.bias', 'sum.18.su.evm.4.weight', 'sum.18.su.evm.4.bias', 'sum.18.su.qpm.0.weight', 'sum.18.su.qpm.0.bias', 'sum.18.su.qpm.2.weight', 'sum.18.su.qpm.2.bias', 'sum.18.su.qpm.4.weight', 'sum.18.su.qpm.4.bias', 'sum.18.su.ppm.0.weight','sum.17.su.nqm.0.weight', 'sum.17.su.nqm.0.bias', 'sum.17.su.nqm.2.weight', 'sum.17.su.nqm.2.bias', 'sum.17.su.nqm.4.weight', 'sum.17.su.nqm.4.bias', 'sum.17.su.eqkm.0.weight', 'sum.17.su.eqkm.0.bias', 'sum.17.su.eqkm.2.weight', 'sum.17.su.eqkm.2.bias', 'sum.17.su.eqkm.4.weight', 'sum.17.su.eqkm.4.bias', 'sum.17.su.epkm.0.weight', 'sum.17.su.epkm.0.bias', 'sum.17.su.epkm.2.weight', 'sum.17.su.epkm.2.bias', 'sum.17.su.epkm.4.weight', 'sum.17.su.epkm.4.bias', 'sum.17.su.evm.0.weight', 'sum.17.su.evm.0.bias', 'sum.17.su.evm.2.weight', 'sum.17.su.evm.2.bias', 'sum.17.su.evm.4.weight', 'sum.17.su.evm.4.bias', 'sum.17.su.qpm.0.weight', 'sum.17.su.qpm.0.bias', 'sum.17.su.qpm.2.weight', 'sum.17.su.qpm.2.bias', 'sum.17.su.qpm.4.weight', 'sum.17.su.qpm.4.bias', 'sum.17.su.ppm.0.weight', 'sum.16.su.nqm.0.weight', 'sum.16.su.nqm.0.bias', 'sum.16.su.nqm.2.weight', 'sum.16.su.nqm.2.bias', 'sum.16.su.nqm.4.weight', 'sum.16.su.nqm.4.bias', 'sum.16.su.eqkm.0.weight', 'sum.16.su.eqkm.0.bias', 'sum.16.su.eqkm.2.weight', 'sum.16.su.eqkm.2.bias', 'sum.16.su.eqkm.4.weight', 'sum.16.su.eqkm.4.bias', 'sum.16.su.epkm.0.weight', 'sum.16.su.epkm.0.bias', 'sum.16.su.epkm.2.weight', 'sum.16.su.epkm.2.bias', 'sum.16.su.epkm.4.weight', 'sum.16.su.epkm.4.bias', 'sum.16.su.evm.0.weight', 'sum.16.su.evm.0.bias', 'sum.16.su.evm.2.weight', 'sum.16.su.evm.2.bias', 'sum.16.su.evm.4.weight', 'sum.16.su.evm.4.bias', 'sum.16.su.qpm.0.weight', 'sum.16.su.qpm.0.bias', 'sum.16.su.qpm.2.weight', 'sum.16.su.qpm.2.bias', 'sum.16.su.qpm.4.weight', 'sum.16.su.qpm.4.bias', 'sum.16.su.ppm.0.weight','sum.26.su.nqm.0.weight', 'sum.26.su.nqm.0.bias', 'sum.26.su.nqm.2.weight', 'sum.26.su.nqm.2.bias', 'sum.26.su.nqm.4.weight', 'sum.26.su.nqm.4.bias', 'sum.26.su.eqkm.0.weight', 'sum.26.su.eqkm.0.bias', 'sum.26.su.eqkm.2.weight', 'sum.26.su.eqkm.2.bias', 'sum.26.su.eqkm.4.weight', 'sum.26.su.eqkm.4.bias', 'sum.26.su.epkm.0.weight', 'sum.26.su.epkm.0.bias', 'sum.26.su.epkm.2.weight', 'sum.26.su.epkm.2.bias', 'sum.26.su.epkm.4.weight', 'sum.26.su.epkm.4.bias', 'sum.26.su.evm.0.weight', 'sum.26.su.evm.0.bias', 'sum.26.su.evm.2.weight', 'sum.26.su.evm.2.bias', 'sum.26.su.evm.4.weight', 'sum.26.su.evm.4.bias', 'sum.26.su.qpm.0.weight', 'sum.26.su.qpm.0.bias', 'sum.26.su.qpm.2.weight', 'sum.26.su.qpm.2.bias', 'sum.26.su.qpm.4.weight', 'sum.26.su.qpm.4.bias', 'sum.26.su.ppm.0.weight', 'sum.27.su.nqm.0.weight', 'sum.27.su.nqm.0.bias', 'sum.27.su.nqm.2.weight', 'sum.27.su.nqm.2.bias', 'sum.27.su.nqm.4.weight', 'sum.27.su.nqm.4.bias', 'sum.27.su.eqkm.0.weight', 'sum.27.su.eqkm.0.bias', 'sum.27.su.eqkm.2.weight', 'sum.27.su.eqkm.2.bias', 'sum.27.su.eqkm.4.weight', 'sum.27.su.eqkm.4.bias', 'sum.27.su.epkm.0.weight', 'sum.27.su.epkm.0.bias', 'sum.27.su.epkm.2.weight', 'sum.27.su.epkm.2.bias', 'sum.27.su.epkm.4.weight', 'sum.27.su.epkm.4.bias', 'sum.27.su.evm.0.weight', 'sum.27.su.evm.0.bias', 'sum.27.su.evm.2.weight', 'sum.27.su.evm.2.bias', 'sum.27.su.evm.4.weight', 'sum.27.su.evm.4.bias', 'sum.27.su.qpm.0.weight', 'sum.27.su.qpm.0.bias', 'sum.27.su.qpm.2.weight', 'sum.27.su.qpm.2.bias', 'sum.27.su.qpm.4.weight', 'sum.27.su.qpm.4.bias', 'sum.27.su.ppm.0.weight', 'sum.28.su.nqm.0.weight', 'sum.28.su.nqm.0.bias', 'sum.28.su.nqm.2.weight', 'sum.28.su.nqm.2.bias', 'sum.28.su.nqm.4.weight', 'sum.28.su.nqm.4.bias', 'sum.28.su.eqkm.0.weight', 'sum.28.su.eqkm.0.bias', 'sum.28.su.eqkm.2.weight', 'sum.28.su.eqkm.2.bias', 'sum.28.su.eqkm.4.weight', 'sum.28.su.eqkm.4.bias', 'sum.28.su.epkm.0.weight', 'sum.28.su.epkm.0.bias', 'sum.28.su.epkm.2.weight', 'sum.28.su.epkm.2.bias', 'sum.28.su.epkm.4.weight', 'sum.28.su.epkm.4.bias', 'sum.28.su.evm.0.weight', 'sum.28.su.evm.0.bias', 'sum.28.su.evm.2.weight', 'sum.28.su.evm.2.bias', 'sum.28.su.evm.4.weight', 'sum.28.su.evm.4.bias', 'sum.28.su.qpm.0.weight', 'sum.28.su.qpm.0.bias', 'sum.28.su.qpm.2.weight', 'sum.28.su.qpm.2.bias', 'sum.28.su.qpm.4.weight', 'sum.28.su.qpm.4.bias', 'sum.28.su.ppm.0.weight', 'sum.29.su.nqm.0.weight', 'sum.29.su.nqm.0.bias', 'sum.29.su.nqm.2.weight', 'sum.29.su.nqm.2.bias', 'sum.29.su.nqm.4.weight', 'sum.29.su.nqm.4.bias', 'sum.29.su.eqkm.0.weight', 'sum.29.su.eqkm.0.bias', 'sum.29.su.eqkm.2.weight', 'sum.29.su.eqkm.2.bias', 'sum.29.su.eqkm.4.weight', 'sum.29.su.eqkm.4.bias', 'sum.29.su.epkm.0.weight', 'sum.29.su.epkm.0.bias', 'sum.29.su.epkm.2.weight', 'sum.29.su.epkm.2.bias', 'sum.29.su.epkm.4.weight', 'sum.29.su.epkm.4.bias', 'sum.29.su.evm.0.weight', 'sum.29.su.evm.0.bias', 'sum.29.su.evm.2.weight', 'sum.29.su.evm.2.bias', 'sum.29.su.evm.4.weight', 'sum.29.su.evm.4.bias', 'sum.29.su.qpm.0.weight', 'sum.29.su.qpm.0.bias', 'sum.29.su.qpm.2.weight', 'sum.29.su.qpm.2.bias', 'sum.29.su.qpm.4.weight', 'sum.29.su.qpm.4.bias', 'sum.29.su.ppm.0.weight', 'sum.30.su.nqm.0.weight', 'sum.30.su.nqm.0.bias', 'sum.30.su.nqm.2.weight', 'sum.30.su.nqm.2.bias', 'sum.30.su.nqm.4.weight', 'sum.30.su.nqm.4.bias', 'sum.30.su.eqkm.0.weight', 'sum.30.su.eqkm.0.bias', 'sum.30.su.eqkm.2.weight', 'sum.30.su.eqkm.2.bias', 'sum.30.su.eqkm.4.weight', 'sum.30.su.eqkm.4.bias', 'sum.30.su.epkm.0.weight', 'sum.30.su.epkm.0.bias', 'sum.30.su.epkm.2.weight', 'sum.30.su.epkm.2.bias', 'sum.30.su.epkm.4.weight', 'sum.30.su.epkm.4.bias', 'sum.30.su.evm.0.weight', 'sum.30.su.evm.0.bias', 'sum.30.su.evm.2.weight', 'sum.30.su.evm.2.bias', 'sum.30.su.evm.4.weight', 'sum.30.su.evm.4.bias', 'sum.30.su.qpm.0.weight', 'sum.30.su.qpm.0.bias', 'sum.30.su.qpm.2.weight', 'sum.30.su.qpm.2.bias', 'sum.30.su.qpm.4.weight', 'sum.30.su.qpm.4.bias', 'sum.30.su.ppm.0.weight', 'sum.31.su.nqm.0.weight', 'sum.31.su.nqm.0.bias', 'sum.31.su.nqm.2.weight', 'sum.31.su.nqm.2.bias', 'sum.31.su.nqm.4.weight', 'sum.31.su.nqm.4.bias', 'sum.31.su.eqkm.0.weight', 'sum.31.su.eqkm.0.bias', 'sum.31.su.eqkm.2.weight', 'sum.31.su.eqkm.2.bias', 'sum.31.su.eqkm.4.weight', 'sum.31.su.eqkm.4.bias', 'sum.31.su.epkm.0.weight', 'sum.31.su.epkm.0.bias', 'sum.31.su.epkm.2.weight', 'sum.31.su.epkm.2.bias', 'sum.31.su.epkm.4.weight', 'sum.31.su.epkm.4.bias', 'sum.31.su.evm.0.weight', 'sum.31.su.evm.0.bias', 'sum.31.su.evm.2.weight', 'sum.31.su.evm.2.bias', 'sum.31.su.evm.4.weight', 'sum.31.su.evm.4.bias', 'sum.31.su.qpm.0.weight', 'sum.31.su.qpm.0.bias', 'sum.31.su.qpm.2.weight', 'sum.31.su.qpm.2.bias', 'sum.31.su.qpm.4.weight', 'sum.31.su.qpm.4.bias', 'sum.31.su.ppm.0.weight', 'spl.sam.0.weight', 'spl.sam.0.bias', 'spl.sam.2.weight', 'spl.sam.2.bias', 'spl.sam.4.weight', 'spl.sam.4.bias', 'spl.zdm.0.weight', 'spl.zdm.0.bias', 'spl.zdm.2.weight', 'spl.zdm.2.bias', 'spl.zdm.4.weight', 'spl.zdm.4.bias', 'spl.zdm_vec.0.weight', 'dm.0.weight', 'dm.0.bias', 'dm.2.weight', 'dm.2.bias', 'dm.4.weight', 'dm.4.bias']

    for name,param in model.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
        

    # reload model
    output_path = "/blue/yanjun.li/riddhishthakare/PeSTo/model/save/i_v4_1_2021-09-07_11-21"  
    model_filepath = os.path.join(output_path, 'model_ckpt.pt')

    train_dat_path = os.path.join(output_path, 'train.dat')
        
    if os.path.isfile(model_filepath) and config_runtime["reload"]:
        logger.print("Reloading model from save file")
        model.load_state_dict(pt.load(model_filepath,  map_location=pt.device('cpu')))
        # get last global step
        global_step = json.loads([l for l in open(train_dat_path, 'r')][-1])['global_step']
        # dynamic positive weight
        pos_ratios = pt.from_numpy(np.array(json.loads([l for l in open(train_dat_path, 'r')][-1])['pos_ratios'])).float().to(device)



    # setup dataloaders
    dataloader_train = setup_dataloader(config_data, '/blue/yanjun.li/riddhishthakare/PeSTo/model/datasets/subunits_train60.txt')
    dataloader_test = setup_dataloader(config_data, '/blue/yanjun.li/riddhishthakare/PeSTo/model/datasets/subunits_test18.txt')
    # dataloader_test9 = setup_dataloader(config_data, '/blue/yanjun.li/riddhishthakare/PeSTo/model/datasets/subunits_test9.txt')
    # debug print
    logger.print(f"> training data size: {len(dataloader_train)}")

    # send model to device
    model = model.to(device)

    pos_weight = pt.tensor([1.5]).float()
    # define losses functions
    criterion = pt.nn.BCEWithLogitsLoss(reduction="none")

    # define optimizer
    optimizer = pt.optim.Adam(model.parameters(), lr=config_runtime["learning_rate"])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # min loss initial value
    min_loss = 1e9

    # quick training step on largest data: memory check and pre-allocation
    batch_data = collate_batch_data([dataloader_train.dataset.get_largest()])
    optimizer.zero_grad()
    losses, _, _ = eval_step(model, device, batch_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)
    loss = pt.sum(losses)
    loss.backward()
    optimizer.step()

    # start training
    for epoch in range(config_runtime['num_epochs']):
        print("Epoch - ", epoch)
        # train mode
        model = model.train()


        # train model
        train_results = []
        for batch_train_data in tqdm(dataloader_train):

            # global step
            global_step += 1

            # set gradient to zero
            optimizer.zero_grad()

            # forward propagation
            losses, y, p = eval_step(model, device, batch_train_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

            # backward propagation
            loss = pt.sum(losses)
            loss.backward()

            # optimization step
            optimizer.step()

            # store evaluation results
            train_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

            # log step
            
            writer.add_scalar("training loss x epoch", loss, epoch)

            with pt.no_grad():
                # scores evaluation results and reset buffer
                scores = scoring(train_results, device=device)
                train_results = []
                # logging(logger, writer, scores, global_step, pos_ratios, "train")
                # save model checkpoint


                model = model.eval()

                # EVALUATION
                #T18

                test_results = []
                for step_te, batch_test_data in enumerate(dataloader_test):
                    # forward propagation
                    losses, y, p = eval_step(model, device, batch_test_data, criterion, pos_ratios, config_runtime['pos_weight_factor'], global_step)

                    # store evaluation results
                    test_results.append([losses.detach().cpu(), y.cpu(), p.cpu()])

                    # stop evaluating
                    if step_te >= config_runtime['eval_size']:
                        break

                scores = scoring(test_results, device=device)

                writer.add_scalar("evaluation loss x epoch", scores['loss'], epoch)

                if min_loss >= scores['loss']:
                    min_loss = scores['loss']


            model = model.train()
        
        if (epoch + 1) % 10 ==0:
            print("Saving model...")
            model_filepath = os.path.join("/blue/yanjun.li/riddhishthakare/PeSTo/model/save/40_SMOL_1.4_semi_8A_real", 'model_ckpt.pt')
            pt.save(model.state_dict(), model_filepath)
        scheduler.step()

    writer.close()


if __name__ == '__main__':
    # train model
    # train(config_data, config_model, config_runtime, '.')
    train1(config_data, config_model, config_runtime, '')
