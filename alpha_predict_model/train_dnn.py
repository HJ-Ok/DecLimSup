import argparse
import os
import pickle
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

warnings.filterwarnings("ignore")
import gc
import json

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim as optim
import wandb
from dataset import *
from model import *
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint
from trainer import *
from transformers.optimization import get_cosine_schedule_with_warmup

from utils import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", required=False)
    parser.add_argument("--fold", type=int, default=0, required=False)
    parser.add_argument("--seed", type=int, default=321, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--batch_size", type=int, default=1024, required=False)
    parser.add_argument("--lr", type=float, default=5e-7, required=False)
    parser.add_argument("--weight_decay", type=float, default=0.01, required=False)
    parser.add_argument("--wandb_name", type=str, default="train", required=False)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


args = parse_args()
print("----args_info----")
print(args)
seed_everything(args.seed)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

job_name = f"Train_{args.wandb_name}"
wandb.init(project=f"TUNA_RM_{args.wandb_name}", config=args, name=job_name)
wandb.config.update(args)

with open("../outputs/Phi-3-mini_gsm8k_train.pkl", "rb") as f:
    train_7b_logit = pickle.load(f)
with open("../outputs/Phi-3-mini_gsm8k_test.pkl", "rb") as f:
    test_7b_logit = pickle.load(f)

with open("../outputs/Phi-3-medium_gsm8k_train.pkl", "rb") as f:
    train_13b_logit = pickle.load(f)
with open("../outputs/Phi-3-medium_gsm8k_test.pkl", "rb") as f:
    test_13b_logit = pickle.load(f)


df = pd.read_csv("../phi3_train_folds.csv")

train_df = df[df.kfold != args.fold]
valid_df = df[df.kfold == args.fold]
test_df = pd.read_csv("../phi3_test.csv")

valid_7b_logit = [train_7b_logit[idx] for idx in valid_df.index.tolist()]
valid_13b_logit = [train_13b_logit[idx] for idx in valid_df.index.tolist()]
train_7b_logit = [train_7b_logit[idx] for idx in train_df.index.tolist()]
train_13b_logit = [train_13b_logit[idx] for idx in train_df.index.tolist()]

device = torch.device("cuda")

train_data_loader = create_data_loader(
    train_7b_logit, train_13b_logit, train_df.reset_index(drop=True), args.batch_size, shuffle_=True
)
valid_data_loader = create_data_loader(
    valid_7b_logit, valid_13b_logit, valid_df.reset_index(drop=True), args.batch_size
)
test_data_loader = create_data_loader(test_7b_logit, test_13b_logit, test_df, args.batch_size)
EPOCHS = args.epochs
model = MLP().to(device)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
total_steps = len(train_data_loader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps
)

max_acc = 0
tmp_max_acc = 0
for epoch in range(EPOCHS):
    print("-" * 10)
    print(f"Epoch {epoch}/{EPOCHS-1}")
    print("-" * 10)
    train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, epoch)

    valid_outputs_arr, valid_label_arr, valid_loss = validate(model, valid_data_loader, device)
    test_outputs_arr, test_label_arr, test_loss = validate(model, test_data_loader, device)
    valid_acc = calc_f1_acc(valid_outputs_arr, valid_df.reset_index(drop=True))
    test_acc = calc_f1_acc(test_outputs_arr, test_df)

    if valid_acc > tmp_max_acc:
        tmp_max_acc = valid_acc
        max_acc = test_acc
    print("valid acc:", valid_acc)
    print("test acc:", test_acc)
    wandb.log(
        {
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
    )

wandb.log({"accuracy": max_acc})
wandb.log({"last_accuracy": test_acc})
wandb.finish()

print(max_acc)
print(test_acc)
