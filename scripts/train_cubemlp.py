"""
This script is used to train a classification model
"""

#%%
import src.convenience
from src.convenience import set_seed
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


#%%
from typing import Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.dataset import Video2Reaction, collate_fn, REACTION_CLASSES
from src.model import MultimodalReactionPredictor
import time
#%%
def get_dataset(cache_folder, metadata_dir, processed_feature_dir, visual_encoder, text_encoder, audio_encoder_acoustic, audio_encoder_semantic, batch_size,
                splits=["train","val","test"]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load the dataset and return the training, validation, and test data loaders
    """
    dataloaders = {}
    for split in splits:
        dataset = Video2Reaction(metadata_file_path=os.path.join(metadata_dir, f"{split}.json"), 
                                 processed_feature_dir=processed_feature_dir, 
                                 visual_encoder=visual_encoder, 
                                 text_encoder=text_encoder, 
                                 audio_encoder_acoustic=audio_encoder_acoustic, 
                                 audio_encoder_semantic=audio_encoder_semantic,
                                 lazy_load=False, use_time_dimension=True,
                                 cache_file_path=f"{cache_folder}/{split}_{visual_encoder}_{text_encoder}_{audio_encoder_acoustic}_{audio_encoder_semantic}.pt")
        dataloaders[split] = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=(split=="train"))
    return dataloaders

#%%
from src.baselines.cubemlp import CubeMLP
def get_model(config) -> nn.Module:
    """
    Load the classification model
    """
    from types import SimpleNamespace
    # Convert the config dictionary to a SimpleNamespace object
    config = SimpleNamespace(**config)
    return CubeMLP(config)

#%%
from typing import Dict

# TODO: compute VA distance matrix
from src.metrics import CumulativeAbsoluteDistanceLoss, CJSLoss, QFD2Loss, compute_va_distance_matrix
from src.dataset import REACTION_CLASSES, REACTION_VA_MATRIX
# Load the distance matrix for QFD2 loss

LOSS_FACTORY = {
    "cross_entropy": nn.CrossEntropyLoss(),
    "kl_divergence": nn.KLDivLoss(reduction='batchmean'),
    "bce": nn.BCEWithLogitsLoss(),
    "mse": nn.MSELoss(),
    "mae": nn.L1Loss(),
    # VA-based Ordinal
    "cad_ordinal_va": CumulativeAbsoluteDistanceLoss(),
    "qfd_ordinal_va": QFD2Loss(),
    "cjs_ordinal_va": CJSLoss(),
    # VA distnace-based
    "qfd_sim_va": QFD2Loss(distance_matrix=compute_va_distance_matrix(REACTION_VA_MATRIX)),
}

def get_loss_fn(loss_type_dict: Dict[str, float]) -> nn.Module:
    """
    Get the loss function based on the specified type
    """
    returned_loss_fn = {}
    for loss_type, weight in loss_type_dict.items():
        if loss_type in LOSS_FACTORY:
            returned_loss_fn[loss_type] = LOSS_FACTORY[loss_type]
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    return returned_loss_fn

#%%
from typing import Dict
from src.metrics import CumulativeAbsoluteDistanceLoss
# Train the model
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                output_type: str,
                checkpoint_dir: str,
                epochs: int, lr: float, loss_type_dict: Dict[str, float] = {"cross_entropy":1.0}, 
                device: str = "cuda",
                random_seed: int = 42,
                ) -> nn.Module:
    """
    Train the model for a given number of epochs and visualize the loss.
    Args:
    - model (nn.Module): The neural network model to train.
    - train_loader (DataLoader): The DataLoader for training data.
    - val_loader (DataLoader): The DataLoader for validation data.
    - epochs (int): Number of training epochs.
    - lr (float): Learning rate for the optimizer.
    - loss_type (str): Choose between 'cross_entropy' or 'kl_divergence' for loss function.
    - plot_path (str): The path to save the loss plot as a PNG file.
    
    Returns:
    - nn.Module: The trained model.
    """

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    # Learning rate scheduler: StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # Loss function setup
    if output_type == "sentiment_distribution":
        num_classes = 4
    else:
        num_classes = len(REACTION_CLASSES) # Total number of possible reactions (class labels) 

    criterion = get_loss_fn(loss_type_dict)

    # Track the best model based on validation loss
    best_model = None
    best_val_metric = float('inf')

    # Store losses for plotting
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Initialize variables for tracking
        running_train_loss = 0.0
        # Set the model to training mode
        model.train()
        
        # Train loop
        for batch_i, batch in enumerate(train_loader):
            # Move data to the same device as the model
            time_start = time.time()
            #print(batch["input"].keys())
            #text_embedding = batch["input"]["clip_description_embedding"].to(device)
            visual_features = batch["input"]["visual_feature"].to(device)
            audio_acoustic_features = batch["input"]["audio_acoustic_feature"].to(device)
            audio_semantic_features = batch["input"]["audio_semantic_feature"].to(device)
            # average of audio features
            audio_feature = (audio_acoustic_features + audio_semantic_features) / 2.0
            #genre_embedding = batch["input"]["movie_genre"].to(device)
            target_distribution = batch["output"][output_type].to(device)



            if DEBUG and batch_i == 0:
                print(f"Time taken to load batch: {time.time() - time_start:.4f} seconds")
            
            # Zero the gradients
            optimizer.zero_grad()
            

            # Forward pass
            time_start = time.time()
            predicted_scores = model(audio_feature, visual_features)
            predicted_distribution = torch.softmax(predicted_scores, dim=-1)
            if DEBUG and batch_i == 0:
                print(f"Time taken for forward pass: {time.time() - time_start:.4f} seconds")
               
            # Calculate loss
            loss = 0.0
            for loss_type, weight in loss_type_dict.items():
                if loss_type == "kl_divergence":
                    # KL Divergence expects both inputs to be log probabilities (softmax/logsoftmax)
                    predicted_log_prob = torch.log_softmax(predicted_distribution, dim=-1)
                    loss += weight * criterion[loss_type](predicted_log_prob, target_distribution)
                elif loss_type == "bce" or loss_type == "cross_entropy":
                    loss += weight * criterion[loss_type](predicted_scores, target_distribution)
                else:
                    # Handle other loss types
                    # BCE + CAD Loss
                    loss += weight * criterion[loss_type](predicted_distribution, target_distribution)

            

            # Backward pass and optimization
            time_start = time.time()
            loss.backward()
            optimizer.step()
            if DEBUG and batch_i == 0:
                print(f"Time taken for backward pass: {time.time() - time_start:.4f} seconds")

            # Track running loss
            running_train_loss += loss.item()

        # Calculate average training loss for this epoch
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # debug
        if DEBUG:
            # check gradient norm
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient norm for {name}: {param.grad.norm()}")
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                visual_features = batch["input"]["visual_feature"].to(device)
                audio_acoustic_features = batch["input"]["audio_acoustic_feature"].to(device)
                audio_semantic_features = batch["input"]["audio_semantic_feature"].to(device)
                # average of audio features
                audio_feature = (audio_acoustic_features + audio_semantic_features) / 2.0
                target_distribution = batch["output"][output_type].to(device)

                # Forward pass
                predicted_scores = model(audio_feature, visual_features)
                
                predicted_distribution = torch.softmax(predicted_scores, dim=-1)
                loss = 0.0
                for loss_type, weight in loss_type_dict.items():
                    if loss_type == "kl_divergence":
                        # KL Divergence expects both inputs to be log probabilities (softmax/logsoftmax)
                        predicted_log_prob = torch.log_softmax(predicted_distribution, dim=-1)
                        loss += weight * criterion[loss_type](predicted_log_prob, target_distribution)
                    elif loss_type == "bce":
                        loss += weight * criterion[loss_type](predicted_scores, target_distribution)
                    else:
                        # BCE + CAD Loss
                        loss += weight * criterion[loss_type](predicted_distribution, target_distribution)


                running_val_loss += loss.item()

        # Calculate average validation loss for this epoch
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Compute evaluation metric as stopping criterion
        val_evaluation_results_distribution, val_evaluation_results_classification = evaluate_model(model, val_loader)

        
        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")
        print("Validation Evaluation Results (Distribution):", val_evaluation_results_distribution)
        print("Validation Evaluation Results (Classification):", val_evaluation_results_classification)

        # Save the best model based on validation loss
        if val_evaluation_results_distribution["mse"] < best_val_metric:
            best_val_metric = val_evaluation_results_distribution["mse"]
            best_model = model.state_dict()
            # Save the best model to disk
            torch.save({
                "epoch": epoch,
                "checkpoint": best_model, 
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "validation_evaluation_results": {
                    "distribution": val_evaluation_results_distribution,
                    "classification": val_evaluation_results_classification
                }
            }, os.path.join(checkpoint_dir, f"best_model_{loss_type_dict}_{random_seed}.pth"))
        # Step the scheduler (update the learning rate)
        scheduler.step()

        # Plot the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(range(epoch+1), train_losses, label="Training Loss", color="blue")
        plt.plot(range(epoch+1), val_losses, label="Validation Loss", color="red")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss (Target: {output_type}, Loss: {loss_type_dict})")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(checkpoint_dir, f"learning_curve_{loss_type_dict}_{random_seed}.png"))
        plt.close()

    # Load the best model found during training
    model.load_state_dict(best_model)

    return model


#%%
from src.metrics import compute_all_classification_metrics, compute_all_distribution_metrics

#%%
from src.dataset import reaction_dist_to_dict, SENTIMENT_2_FINER_GRAINED_MAPPING, REACTION_CLASSES

def map_emotion_to_sentiment(emotion: str) -> str:
    """
    Map a finer-grained emotion to its corresponding sentiment.
    """
    for sentiment, emotions in SENTIMENT_2_FINER_GRAINED_MAPPING.items():
        if emotion in emotions:
            return sentiment
    raise ValueError(f"Emotion '{emotion}' not found in SENTIMENT_2_FINER_GRAINED_MAPPING.")



def evaluate_model(model: nn.Module, test_loader: DataLoader, output_type="reaction_distribution", device="cuda", save_pred=False, pred_path=None) -> Tuple[float, float]:
    """
    Evaluate the model
    """
    model.eval()  # Set model to evaluation mode
    video_ids = []
    predicted_outputs = []
    target_outputs = []

    with torch.no_grad():
        for batch in test_loader:
            video_ids.extend(batch["video_id"])
            visual_features = batch["input"]["visual_feature"].to(device)
            audio_acoustic_features = batch["input"]["audio_acoustic_feature"].to(device)
            audio_semantic_features = batch["input"]["audio_semantic_feature"].to(device)
            # average of audio features
            audio_feature = (audio_acoustic_features + audio_semantic_features) / 2.0
            target_labels = batch["output"][output_type].to(device)
            print(target_labels.shape)

            # Forward pass
            predicted_scores = model(audio_feature, visual_features)
            predicted_distribution = torch.softmax(predicted_scores, dim=-1)
            print(predicted_distribution.shape)
            predicted_outputs.append(predicted_distribution.detach().cpu().numpy())
            target_outputs.append(target_labels.detach().cpu().numpy())

    predicted_outputs = np.concatenate(predicted_outputs, axis=0)
    target_outputs = np.concatenate(target_outputs, axis=0)
    if save_pred:
        np.save(pred_path, predicted_outputs)

    # Compute evaluation metrics
    distribution_results = compute_all_distribution_metrics(predicted_outputs, target_outputs)
    classification_results = compute_all_classification_metrics(predicted_outputs, target_outputs)
    
    return distribution_results, classification_results

    
#%%


class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#%%
# python train_classification.py
def main():
    """
    Train a classification model
    """
    parser = argparse.ArgumentParser(description="Train a classification model")
    # to load data
    parser.add_argument("--cache_folder")
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Directory containing the metadata files")
    parser.add_argument("--processed_feature_dir", type=str, default="data/processed_features", help="Directory containing the processed features")
    parser.add_argument("--visual_encoder", type=str, default="vivit", help="Type of visual encoder")
    parser.add_argument("--text_encoder", type=str, default="bert-base-uncased", help="Name of the Hugging Face BERT model")
    parser.add_argument("--audio_encoder_acoustic", type=str, default="clap_general", help="Type of audio encoder for acoustic features")
    parser.add_argument("--audio_encoder_semantic", type=str, default="hubert_large", help="Type of audio encoder for semantic features")
    parser.add_argument("--use_time_dimension", action="store_true", help="Use time dimension in the model")

    # to train model
    parser.add_argument("--config_file", type=str, default="config.json", help="Path to the configuration file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--loss_type_dict", type=str, default="cross_entropy", help="Loss function to use")
    parser.add_argument("--reaction_outcome_type", type=str, default="reaction_distribution", help="Type of reaction outcome to predict")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    # to save model
    parser.add_argument("--checkpoint_base_dir", type=str, default="models", help="Directory to save the trained model")
    # to evaluate model
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--test", action="store_true", help="Evaluate the model on the test set")

    args = parser.parse_args()

    args.loss_type_dict = eval(args.loss_type_dict)
    # Generate all possible loss_type_dict
    ALL_LOSS_TYPE_DICT = []
    #%%
    os.makedirs(args.checkpoint_base_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_base_dir

    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    set_seed(args.random_seed)

    #%%
    dataloaders = get_dataset(cache_folder = args.cache_folder,
        metadata_dir=args.metadata_dir,
        processed_feature_dir=args.processed_feature_dir,
        visual_encoder=args.visual_encoder,
        text_encoder=args.text_encoder,
        audio_encoder_acoustic=args.audio_encoder_acoustic,
        audio_encoder_semantic=args.audio_encoder_semantic,
        batch_size=args.batch_size,
        splits=["train", "val", "test"])

    #%%
    model_config = json.load(open(args.config_file))

    model = get_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #%%
    if args.train:
        #%%
        model = train_model(model, dataloaders["train"], dataloaders["val"], args.reaction_outcome_type,
        checkpoint_dir, args.epochs, args.lr, args.loss_type_dict, random_seed=args.random_seed, device=device)
    
    #%%
    if args.test:
        #%%
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"best_model_{args.loss_type_dict}_{args.random_seed}.pth"))["checkpoint"])
        model.to(device)
        #%%
        # Run evaluation on train
        train_evaluation_results  = evaluate_model(model, dataloaders["train"], args.reaction_outcome_type)
        # Save the evaluation results to a JSON file
        results_path = os.path.join(checkpoint_dir, f"train_evaluation_result_{args.loss_type_dict}_{args.random_seed}.json")
        with open(results_path, "w") as f:
            json.dump(train_evaluation_results, f, indent=2, cls=NumpyTypeEncoder)
        logging.info(f"Train evaluation results saved to {results_path}")
        
        #%%
        # Run evluation on validation
        val_evaluation_results = evaluate_model(model, dataloaders["val"], args.reaction_outcome_type)
        # Save the evaluation results to a JSON file

        results_path = os.path.join(checkpoint_dir, f"val_evaluation_result_{args.loss_type_dict}_{args.random_seed}.json")
        with open(results_path, "w") as f:
            json.dump(val_evaluation_results, f, indent=2, cls=NumpyTypeEncoder)
        logging.info(f"Validation evaluation results saved to {results_path}")
        
        #%%
        # Run evaluation on test
        test_evaluation_metrics = evaluate_model(model, dataloaders["test"], args.reaction_outcome_type, save_pred=True, 
                                                 pred_path=os.path.join(checkpoint_dir, f"test_pred_{args.loss_type_dict}_{args.random_seed}.npy"))
        test_evaluation_results = {
            "model_config": model_config,
            "loss_type_dict": args.loss_type_dict,
            "random_seed": args.random_seed,
            "test_evaluation_metrics": test_evaluation_metrics
        }
        #%%
        # # Generate results for each sample
        # sample_results_path = os.path.join(checkpoint_dir, f"{args.loss_type_dict}_test_generation.json")
        # with open(sample_results_path, "w") as f:
        #     json.dump(test_sample_results, f, indent=2, cls=NumpyTypeEncoder)
        #%%
        # Save the evaluation results to a JSON file
        results_path = os.path.join(checkpoint_dir, f"test_evaluation_result_{args.loss_type_dict}_{args.random_seed}.json")
        with open(results_path, "w") as f:
            json.dump(test_evaluation_results, f, indent=2, cls=NumpyTypeEncoder)
        logging.info(f"Test evaluation results saved to {results_path}")
    
# %%
# python train_classification.py 
if __name__ == "__main__":
    main()
