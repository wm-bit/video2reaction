"""
This script is used to train a classification model
"""

#%%
import src.convenience
import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List
import json

from typing import Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import Video2Reaction, collate_fn, REACTION_CLASSES
from pyldl.algorithms import LDSVR, SA_BFGS, SA_IIS, AA_KNN, PT_Bayes, GLLE, LIBLE, PT_SVM, LDL_LRR, LDL_HVLC, TLRLDL
from pyldl.metrics import score

from sklearn.preprocessing import StandardScaler
from src.metrics import compute_all_classification_metrics, compute_all_distribution_metrics
import time

import random
import tensorflow as tf
#%%
def get_dataset(cache_folder, metadata_dir, processed_feature_dir, visual_encoder, text_encoder, audio_encoder_semantic, audio_encoder_acoustic, batch_size,
                splits=["train", "test"], included_input_features = ["clip_description_embedding", "visual_feature", "audio_acoustic_feature", "audio_semantic_feature", "movie_genre"]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the dataset and return the training, validation, and test data loaders
    """
    dataloaders = {}
    for split in splits:
        dataset = Video2Reaction(os.path.join(metadata_dir, f"{split}.json"), processed_feature_dir, 
                                 visual_encoder, text_encoder, 
                                 lazy_load=False,
                                 cache_file_path=f"{cache_folder}/{split}_{visual_encoder}_{text_encoder}_{audio_encoder_acoustic}_{audio_encoder_semantic}_no_time_dimension.pt", 
                                 use_time_dimension=False, 
                                 input_type="processed",
                                 device="cpu")
        # return X_train, y_train, X_val, y_val, X_test, y_test in numpy array
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True if split == "train" else True, collate_fn=collate_fn)
        train_X = []
        train_Y = []
        for batch in dataloader:
            combined_input = torch.cat([m.to("cpu") for k, m in batch["input"].items() if k in included_input_features], dim=1)
            train_X.append(combined_input.numpy())
            train_Y.append(batch["output"]["reaction_distribution"].numpy())
        train_X = np.concatenate(train_X, axis=0)
        train_Y = np.concatenate(train_Y, axis=0)
        dataloaders[split] = (train_X, train_Y)
    return dataloaders



#%%
class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

import itertools

def generate_hyperparam_settings(hyperparams):
    # Extract the keys and values from the hyperparams dictionary
    keys = hyperparams.keys()
    values = hyperparams.values()

    # Create a list of all combinations of hyperparameters
    hyperparam_combinations = list(itertools.product(*values))

    # Convert the combinations into a list of dictionaries
    hyperparam_settings = [dict(zip(keys, combination)) for combination in hyperparam_combinations]

    return hyperparam_settings
#%%
import pickle
from pyldl.utils import binaryzation

#%%
if __name__ == "__main__":
    # Load dataset
    #%%
    parser = argparse.ArgumentParser(description="Run baselines")
    parser.add_argument("--cache_folder", type=str, default="../data/cache/", help="Directory to save the cache files")
    parser.add_argument("--metadata_dir", type=str, default="../data/metadata/")
    parser.add_argument("--processed_feature_dir", type=str, default="../data/processed_features/")
    parser.add_argument("--visual_encoder", type=str, default="clip")
    parser.add_argument("--text_encoder", type=str, default="clip")
    parser.add_argument("--audio_encoder_semantic", type=str, default=None)
    parser.add_argument("--audio_encoder_acoustic", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--checkpoint_base_dir", type=str, default="models", help="Directory to save the trained model")
    parser.add_argument("--included_input_features", type=str, nargs="+", default=["clip_description_embedding", "visual_feature", "audio_acoustic_feature", "audio_semantic_feature", "movie_genre"], help="List of input features to include")
    # to evaluate model
    parser.add_argument("--reprocess", action="store_true", help="Reprocess the data")
    parser.add_argument("--test_only", action="store_true", help="Only test the model")
    # seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # method
    parser.add_argument("--methods", type=str, nargs="+", help="List of methods to run")

    args = parser.parse_args()
    #%%
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    #%%
    os.makedirs(args.checkpoint_base_dir, exist_ok=True)
    checkpoint_paths = []
    if "visual_feature" in args.included_input_features:
        checkpoint_paths.append(args.visual_encoder)
    if "clip_description_embedding" in args.included_input_features:
        checkpoint_paths.append(args.text_encoder)
    if "audio_acoustic_feature" in args.included_input_features:
        checkpoint_paths.append(args.audio_encoder_acoustic)
    if "audio_semantic_feature" in args.included_input_features:
        checkpoint_paths.append(args.audio_encoder_semantic)
    if "movie_genre" in args.included_input_features:
        checkpoint_paths.append("movie_genre")
    checkpoint_paths = "_".join(checkpoint_paths)
    checkpoint_dir = os.path.join(args.checkpoint_base_dir, f"{checkpoint_paths}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    #methods = ['LDSVR', 'SA_BFGS', 'PT_Bayes', "LDL_LRR", "TLRLDL"]
    methods = args.methods
    # TODO: need to add methods that requires preprocessing and hyperparam tuning (LDL_LRR, LDL_HVLC)
    
    #%%
    print("Loading dataset...")
    if args.test_only:
        # Check if test_Y.npy exists
        if not os.path.exists(os.path.join(checkpoint_dir, "test_Y.npy")):
            dataset = get_dataset(args.cache_folder, args.metadata_dir, args.processed_feature_dir, args.visual_encoder, args.text_encoder,
                                  args.audio_encoder_acoustic, args.audio_encoder_semantic,
                                  batch_size=args.batch_size,
                                  splits=["val", "test"], included_input_features=args.included_input_features)
            test_X, test_Y = dataset["test"]
            val_X, val_Y = dataset["val"]
        else:
            test_Y = np.load(os.path.join(checkpoint_dir, "test_Y.npy"))
            test_X = None
    else:
        dataset = get_dataset(args.metadata_dir, args.processed_feature_dir, args.visual_encoder, args.text_encoder, 
                            audio_encoder_acoustic=args.audio_encoder_acoustic, audio_encoder_semantic=args.audio_encoder_semantic,
                            batch_size=args.batch_size,
                            splits=["train", "val", "test"], included_input_features=args.included_input_features)
        train_X, train_Y = dataset["train"]
        test_X, test_Y = dataset["test"]
        val_X, val_Y = dataset["val"]

        # Save test_Y
        if not os.path.exists(os.path.join(checkpoint_dir, "test_Y.npy")):
            print("Saving test_Y...")
            # Save test_Y
            np.save(os.path.join(checkpoint_dir, "test_Y.npy"), test_Y)
        if not os.path.exists(os.path.join(checkpoint_dir, "val_Y.npy")):
            np.save(os.path.join(checkpoint_dir, "val_Y.npy"), val_Y)

    #%%
    print("Training and evaluating models...")
    baseline_results = {}
    # if os.path.exists(os.path.join(checkpoint_dir, "baseline_results.json")) and not args.reprocess:
    #     print(f"Loading baseline results from {checkpoint_dir}...")
    #     with open(os.path.join(checkpoint_dir, "baseline_results.json"), "r") as f:
    #         baseline_results = json.load(f)
    for method in methods:
        print(f"Method: {method}")
        if os.path.exists(os.path.join(checkpoint_dir, f"{method}_{args.seed}_pred.npy")) and not args.reprocess:
            print(f"Loading predictions for {method}...")
            Y_pred = np.load(os.path.join(checkpoint_dir, f"{method}_{args.seed}_pred.npy"))
        else:
            # Test methods with hyperparameters
            hyperparams = {}
            if method in  ["LDL_LRR", "TLRLDL"]:
                hyperparams = {
                    "alpha": [1e-2, 1e-1, 1.0, 10.0],
                    "beta": [1e-2, 1e-1, 1.0, 10.0],
                }
            elif method == "LDL_HVLC":
                hyperparams = {
                    "k": [5, 10, 15, 20]
                }
            # Transform data 
            if method in ["LDL_LRR", "LDL_HVLC", "TLRLDL"]:
                print("Standardizing data before training...")
                scaler = StandardScaler()
                scaler.fit(train_X)
                train_X = scaler.transform(train_X)
                test_X = scaler.transform(test_X)
            if method in ["TLRLDL"]:
                # transform Y into multilabel
                train_Y = binaryzation(train_Y, method="threshold", param=0.5)
                #val_Y = binaryzation(val_Y, method="threshold", param=0.5)

            if len(hyperparams) > 0:
                # Only do grid search for one random seed
                if os.path.exists(os.path.join(checkpoint_dir, f"{method}_best_hyperparam.json")) and not args.reprocess:
                    print(f"Loading best hyperparameter setting for {method}...")
                    with open(os.path.join(checkpoint_dir, f"{method}_best_hyperparam.json"), "r") as f:
                        best_hyperparam_setting = json.load(f)
                    all_hyperparam_settings = [best_hyperparam_setting]
                    grid_search = False
                else:
                    # Grid search for hyperparameters
                    grid_search = True
                    all_hyperparam_settings = generate_hyperparam_settings(hyperparams)
                    print(f"Number of hyperparameter settings: {len(all_hyperparam_settings)}")
                best_val_score = float("inf")
                best_model = None
                for hyperpram_setting in all_hyperparam_settings:
                    hyperparam_str = "_".join([f"{k}_{v}" for k, v in hyperpram_setting.items()])
                    print(f"Hyperparameter setting: {hyperparam_str}")
                    # Create a directory for the hyperparameter setting
                    hyperparam_pred_path = os.path.join(checkpoint_dir, f"{method}_{hyperparam_str}_{args.seed}.pt")
                    if os.path.exists(hyperparam_pred_path) and not args.reprocess:
                        print(f"Loading model for {hyperparam_str}...")
                        model = pickle.load(open(hyperparam_pred_path, "rb"))
                    else:
                        model = eval(method)(**hyperpram_setting, random_state=args.seed)
                        model.fit(train_X, train_Y)
                        # # Save model
                        # with open(hyperparam_pred_path, "wb") as f:
                        #     pickle.dump(model, f)
                    # Check validation set performance
                    val_Y_pred = model.predict(val_X)
                    print(f"Validation set performance: {val_Y_pred.shape}")
                    val_distribution_metrics = compute_all_distribution_metrics(val_Y, val_Y_pred)
                    metric_to_select = "kl"
                    if val_distribution_metrics[metric_to_select] < best_val_score:
                        best_val_score = val_distribution_metrics[metric_to_select]
                        best_hyperparam_setting = hyperpram_setting
                        # Save best model
                        best_model = model
                    
                    print(f"Best hyperparameter setting: {best_hyperparam_setting}")
                    # Save hyperparameter setting
                if grid_search:
                    with open(os.path.join(checkpoint_dir, f"{method}_best_hyperparam.json"), "w") as f:
                        json.dump(best_hyperparam_setting, f, indent=4, cls=NumpyTypeEncoder)
            
                
                Y_pred = best_model.predict(test_X)
                # Save best model
                np.save(os.path.join(checkpoint_dir, f"{method}_{args.seed}_pred.npy"), Y_pred)
                # Save val
                val_Y_pred = best_model.predict(val_X)
                np.save(os.path.join(checkpoint_dir, f"{method}_{args.seed}_val_pred.npy"), val_Y_pred)
            else:
                model = eval(method)()
                model.fit(train_X, train_Y)
                Y_pred = model.predict(test_X)
                # Save Y_pred
                np.save(os.path.join(checkpoint_dir, f"{method}_{args.seed}_pred.npy"), Y_pred)
                # Save val
                val_Y_pred = model.predict(val_X)
                np.save(os.path.join(checkpoint_dir, f"{method}_{args.seed}_val_pred.npy"), val_Y_pred)
        #%%
        discriminative_metrics = compute_all_classification_metrics(test_Y, Y_pred)
        distribution_metrics = compute_all_distribution_metrics(test_Y, Y_pred)
        baseline_results[method] = {
            "discriminative_metrics": discriminative_metrics,
            "distribution_metrics": distribution_metrics
        }
        # Save results
        with open(os.path.join(checkpoint_dir, f"{method}_{args.seed}_results.json"), "w") as f:
            json.dump(baseline_results[method], f, indent=4, cls=NumpyTypeEncoder)
        # Run on validation set
        val_discriminative_metrics = compute_all_classification_metrics(val_Y, val_Y_pred)
        val_distribution_metrics = compute_all_distribution_metrics(val_Y, val_Y_pred)
        with open(os.path.join(checkpoint_dir, f"{method}_{args.seed}_val_results.json"), "w") as f:
            json.dump({
                "discriminative_metrics": val_discriminative_metrics,
                "distribution_metrics": val_distribution_metrics
            }, f, indent=4, cls=NumpyTypeEncoder)




# %%
