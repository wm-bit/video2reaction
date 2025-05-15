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
import engine.mminfomax as mminfomax_engine


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
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--lr", type=float, required=False)
    parser.add_argument("--num_epochs", type=int, required=False)
    parser.add_argument("--test_mod", type=bool, required=False, default=False)

    parser.add_argument('--dropout_a', type=float, default=0.1,
                        help='dropout of acoustic LSTM out layer')
    parser.add_argument('--dropout_v', type=float, default=0.1,
                        help='dropout of visual LSTM out layer')
    parser.add_argument('--dropout_prj', type=float, default=0.1,
                        help='dropout of projection layer')

    # # Architecture
    # parser.add_argument('--multiseed', action='store_true', help='training using multiple seed')
    # parser.add_argument('--contrast', action='store_true', help='using contrast learning')
    # parser.add_argument('--add_va', type=bool, default=True, action='store_true', help='if add va MMILB module')
    parser.add_argument('--n_layer', type=int, default=1,
                        help='number of layers in LSTM encoders (default: 1)')
    parser.add_argument('--cpc_layers', type=int, default=1,
                        help='number of layers in CPC NCE estimator (default: 1)')
    # parser.add_argument('--d_vh', type=int, default=16,
    #                     help='hidden size in visual rnn')
    # parser.add_argument('--d_ah', type=int, default=16,
    #                     help='hidden size in acoustic rnn')

    parser.add_argument('--d_vout', type=int, default=16,
                        help='output size in visual rnn')
    parser.add_argument('--d_aout', type=int, default=16,
                        help='output size in acoustic rnn')
    parser.add_argument('--d_tout', type=int, default=16,
                        help='output size in visual rnn')
    # parser.add_argument('--d_vout', type=int, default=64,
    #                     help='output size in visual rnn')
    # parser.add_argument('--d_aout', type=int, default=64,
    #                     help='output size in acoustic rnn')
    # parser.add_argument('--d_tout', type=int, default=64,
    #                     help='output size in visual rnn')
    
    parser.add_argument('--d_tfeatdim', type=int, default=768,
                        help='output size in visual rnn')
    parser.add_argument('--d_afeatdim', type=int, default=1024,
                        help='output size in acoustic rnn')
    parser.add_argument('--d_vfeatdim', type=int, default=768,
                        help='output size in visual rnn')
    
    parser.add_argument('--n_class', type=int, default=21,
                        help='label space')
    
    # parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional rnn')
    parser.add_argument('--d_prjh', type=int, default=128,
                        help='hidden size in projection network')
    parser.add_argument('--pretrain_emb', type=int, default=768,
                        help='dimension of pretrained model output')


    # Activations
    parser.add_argument('--mmilb_mid_activation', type=str, default='ReLU',
                        help='Activation layer type in the middle of all MMILB modules')
    parser.add_argument('--mmilb_last_activation', type=str, default='Tanh',
                        help='Activation layer type at the end of all MMILB modules')
    parser.add_argument('--cpc_activation', type=str, default='Tanh',
                        help='Activation layer type in all CPC modules')
        
    parser.add_argument('--update_batch', type=int, default=1,
                        help='update batch interval')

    # Logistics
    parser.add_argument('--log_interval', type=int, default=100,
                        help='frequency of result logging (default: 100)')
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
    
    if args.audio_mode:
        cfg["train"]["audio_mode"] = args.audio_mode
    
    if args.loss:
        cfg["train"]["loss_fn"] = args.loss

    exp_name += "_{}_{}".format(cfg["train"]["audio_mode"], cfg["train"].get("loss_fn", "CrossEntropyLoss"))

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
    
    if args.num_epochs:
        cfg["train"]["num_epochs"] = args.num_epochs

    report_path = os.path.join(ROOT_PATH, "results", exp_name)

    if not os.path.exists(report_path):
        os.makedirs(report_path)

    """
    get top k from prev experiment
    """
    if args.test_mod:
         p = mminfomax_engine.InfoMax(cfg, report_path, model_args=args, test_mod=True)
    else:
        p = mminfomax_engine.InfoMax(cfg, report_path, model_args=args)
    p.run()
    p.test()

