import time

import numpy as np
import torch
import torch.nn as nn

from utils import AverageMeter, calc_f1_acc, timeSince


def round_to_quarter_tensor(numbers):
    clamped_numbers = torch.clamp(numbers, min=-1.0, max=3.0)
    rounded_numbers = torch.round(clamped_numbers / 0.25) * 0.25
    return rounded_numbers


def train_epoch(model, data_loader, optimizer, device, scheduler, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    sent_count = AverageMeter()

    start = end = time.time()

    model = model.train()
    correct_predictions = 0
    for step, d in enumerate(data_loader):
        data_time.update(time.time() - end)
        batch_size = d["input_tensor"].size(0)

        input_tensor = d["input_tensor"].to(device)
        labels = d["labels"].to(device)

        _, loss = model(
            input_tensor=input_tensor,
            labels=labels,
        )

        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

    return correct_predictions / (step + 1), losses.avg


def validate(model, data_loader, device):
    model = model.eval()
    losses = []
    cnt = 0
    for d in data_loader:
        with torch.no_grad():
            input_tensor = d["input_tensor"].to(device)
            labels = d["labels"].to(device)

            outputs, loss = model(
                input_tensor=input_tensor,
                labels=labels,
            )

            losses.append(loss.item())
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if cnt == 0:
                cnt += 1
                outputs_arr = outputs
                label_arr = labels
            else:
                outputs_arr = torch.cat([outputs_arr, outputs], 0)
                label_arr = torch.cat([label_arr, labels], 0)
    return outputs_arr, label_arr, np.mean(losses)
