import sys, os

base_dir = os.getcwd()
sys.path.insert(0, base_dir)

import model
import importlib
import numpy as np
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
import copy
import yaml
import ctools
from easydict import EasyDict as edict
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from torch.utils.tensorboard import SummaryWriter
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config):
    # ===============================> Setup <================================

    dataloader = importlib.import_module("reader." + config.reader)
    # torch.cuda.set_device(config.device)
    cudnn.benchmark = True

    data = config.data
    save = config.save
    params = config.params
    average_loss_num = config.ave_loss

    writer = SummaryWriter()

    print("========================> Read Data <========================")
    if data.isFolder:
        data, _ = ctools.readfolder(data)

    dataset = dataloader.loader(data, params.batch_size, shuffle=True, num_workers=8)

    save_path = os.path.join(save.metapath, save.folder, "checkpoint")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("========================> Model Building <========================")
    net = model.Model()
    net = net.to(device)
    net.train()

    # Pretrain
    # pretrain = config.pretrain
    # if pretrain.enable and pretrain.device:
    #     net.load_state_dict(
    #         torch.load(pretrain.path, map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"}))
    # elif pretrain.enable and not pretrain.device:
    #     net.load_state_dict(torch.load(pretrain.path))

    print("========================> Optimizer Building <========================")
    optimizer = optim.Adam(net.parameters(), lr=params.lr, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=params.decay_step, gamma=params.decay)

    if params.warmup:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=params.warmup,
                                           after_scheduler=scheduler)

    # =======================================> Training < ==========================
    print("========================> Training <========================")
    length = len(dataset)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)

    criterion = nn.L1Loss()
    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    with open(os.path.join(save_path, "train_log"), "w") as outfile:
        outfile.write(ctools.DictDumps(config) + "\n")

        running_loss = 0.0

        for epoch in range(1, params.epoch + 1):
            for i, (data, label) in enumerate(dataset):

                # ------------------forward--------------------
                # data["face"] = data["face"].to(device)
                for key in data:
                    if key != 'name':
                        data[key] = data[key].to(device)
                label = label.to(device)

                out = net(data['face']).to(device)
                loss = criterion(out, label).to(device)

                # loss = net.module.loss(data, label).cuda()
                # loss = net.loss(data, label)

                # -----------------backward--------------------
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                running_loss += loss.item()

                rest = timer.step() / 3600

                # -----------------loger----------------------
                if i % 20 == 0:
                    log = (
                            f"[{epoch}/{params.epoch}]: "
                            + f"[{i}/{length}] "
                            + f"loss:{loss} "
                            + f"lr:{ctools.GetLR(optimizer)} "
                            + f"rest time:{rest:.2f}h"
                    )

                    print(log)
                    outfile.write(log + "\n")
                    sys.stdout.flush()
                    outfile.flush()

                    # Calculate the average loss, you should modify average_loss_num in config file.
                if i % average_loss_num == average_loss_num - 1:
                    writer.add_scalar(
                        'Train loss',
                        running_loss / average_loss_num,
                        epoch * len(dataset) + i
                    )
                    running_loss = 0.0

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{save.model_name}.pt"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch Basic Model Training")

    parser.add_argument("-c", "--config", type=str, help="The source config for training.")

    args = parser.parse_args()

    config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    train_config = config.train

    print("=====================>> (Begin) Training params << =======================")

    print(ctools.DictDumps(train_config))

    print("=====================>> (End) Training params << =======================")

    main(train_config)
