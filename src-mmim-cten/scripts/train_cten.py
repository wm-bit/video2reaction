import sys
sys.path.append(".")

import datetime

import yaml
import os
import torch
import numpy as np
import random
import argparse
import json
import engine.cten as cten_engine


ROOT_PATH = "/work/v2r"


if __name__ == "__main__":
    """
    main.py is only the entrance to the pipeline
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument("--seed_idx", type=int, required=True)
    parser.add_argument('--exp_version', required=False)
    parser.add_argument('--audio_mode', required=False, default="acoustic")
    parser.add_argument("--loss", type=str, required=False)
    parser.add_argument('--is_erasing', type=int, required=False, default=None)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--test_mod", type=bool, required=False, default=False)
    args = parser.parse_args()


    # lock all random seed to make the experiment replicable
    RANDOM_SEEDS = [320, 728, 193, 846, 502]
    seed = RANDOM_SEEDS[args.seed_idx]

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with open(os.path.join('./config/v2r', args.config), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    exp_name = cfg["exp_name"]
    # print(args.is_erasing)
    if args.is_erasing is not None:
        cfg["train"]["is_erasing"] = (args.is_erasing != 0)
    
    if args.audio_mode:
        cfg["train"]["audio_mode"] = args.audio_mode
    
    if args.loss:
        cfg["train"]["loss_fn"] = args.loss

    exp_name += "_{}_{}".format(cfg["train"]["audio_mode"], cfg["train"].get("loss_fn", "CrossEntropyLoss"))

    if cfg["train"]["is_erasing"]:
        exp_name += "_erase"

    if args.exp_version:
        exp_name += "_{}".format(args.exp_version)
    
    if args.test_mod:
        exp_name += "_testmod"
    
    exp_name += f"_seed{seed}"

    cfg["exp_name"] = exp_name

    if args.batch_size:
        cfg["train"]["batch_size"] = args.batch_size

    if args.lr:
        cfg["train"]["lr"] = args.lr

    report_path = os.path.join(ROOT_PATH, "results", exp_name)

    if not os.path.exists(report_path):
        os.makedirs(report_path)

    """
    get top k from prev experiment
    """
    if args.test_mod:
         p = cten_engine.VAANetErase(cfg, report_path, test_mod=True)
    else:
        p = cten_engine.VAANetErase(cfg, report_path)
    p.run()
    p.test()


