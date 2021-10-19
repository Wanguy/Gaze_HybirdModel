import os, sys

base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main(train, test):
    # ===============================> Setup <============================
    reader = importlib.import_module("reader." + test.reader)

    data = test.data
    load = test.load
    # torch.cuda.set_device(test.device)
    writer = SummaryWriter()

    # ==============================> Read Data <========================
    print("========================> Read Data <========================")
    if data.isFolder:
        data, _ = ctools.readfolder(data)

    print(f"========================> Test: {data.label} <========================")
    dataset = reader.loader(data, 32, num_workers=4, shuffle=False)

    model_path = os.path.join(train.save.metapath,
                              train.save.folder, 'checkpoint')
    log_path = os.path.join(train.save.metapath,
                            train.save.folder, f'{test.savename}')

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # =============================> Test <==============================
    print("========================> Testing <========================")

    begin = load.begin_step
    end = load.end_step
    step = load.steps

    for saveiter in range(begin, end + step, step):
        print(f"Test {saveiter}")

        # ----------------------Load Model------------------------------
        net = model.Model()
        # net = nn.DataParallel(net, device_ids=[0, 1, 2])

        state_dict = torch.load(
            os.path.join(model_path, f"Iter_{saveiter}_{train.save.model_name}.pt"),
            map_location={f"cuda:{train.device}": f"cuda:{test.device}"}
        )

        net = net.to(device)
        net.load_state_dict(state_dict)
        net.eval()

        # length = len(dataset)
        accs = 0
        count = 0

        # -----------------------Open log file--------------------------------
        log_name = f"{saveiter}.log"

        outfile = open(os.path.join(log_path, log_name), 'w')
        outfile.write("name results ground_truths\n")

        # -------------------------Testing---------------------------------
        with torch.no_grad():

            for j, (data, label) in enumerate(dataset):

                for key in data:
                    if key != 'name':
                        data[key] = data[key].to(device)

                names = data["name"]

                ground_truths = label.to(device)
                gazes = net(data['face'])

                for k, gaze in enumerate(gazes):
                    gaze = gaze.cpu().detach().numpy()
                    ground_truth = ground_truths.cpu().numpy()[k]

                    count += 1
                    accs += gtools.angular(
                        gtools.gazeto3d(gaze),
                        gtools.gazeto3d(ground_truth)
                    )

                    name = [names[k]]
                    gaze = [str(_) for _ in gaze]
                    ground_truth = [str(_) for _ in ground_truth]
                    log = name + [",".join(gaze)] + [",".join(ground_truth)]
                    outfile.write(" ".join(log) + "\n")

            writer.add_scalars(
                'Avg',
                {'valid': accs / count},
                saveiter
            )
            loger = f"[{saveiter}] Total Num: {count}, avg: {accs / count}"
            outfile.write(loger)
            print(loger)
            writer.add_graph(net, data['face'])
        outfile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch Basic Model Training")

    parser.add_argument("-c", "--config", type=str, help="The source config for training.")

    args = parser.parse_args()
    # Read model from train config and Test data in test config.
    config = edict(yaml.load(open(args.config), Loader=yaml.FullLoader))

    train_config = config.train

    test_config = config.test

    print("=======================>(Begin) Config of training<======================")

    print(ctools.DictDumps(train_config))

    print("=======================>(End) Config of training<======================")

    print("")

    print("=======================>(Begin) Config for test<======================")

    print(ctools.DictDumps(test_config))

    print("=======================>(End) Config for test<======================")

    main(train_config, test_config)
