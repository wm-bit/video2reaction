"""
This script is used to perform reaciton prediiction using VLM (Vision-Language Model).
"""


#%%
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
from tqdm import tqdm
import random

#%%
import src.convenience
from typing import Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from src.dataset import Video2Reaction, collate_fn, REACTION_CLASSES
from src.model import ReactionPredictionModel
import time
from src.zoo import VLM_ZOO
#%%
def get_dataset(metadata_dir, key_frame_dir, split) -> Dataset:
    """
    Load the dataset and return the training, validation, and test data loaders
    """
    # TODO: Do batch inference for VLM, add dataloader
    dataset = Video2Reaction(os.path.join(metadata_dir, f"{split}.json"), input_type="vlm", key_frame_dir=key_frame_dir, 
                             lazy_load=True)
    return dataset


def get_model(model_name: str) -> nn.Module:
    """
    Load the VLM
    """
    from transformers import AutoProcessor, AutoModelForPreTraining
    model_path = VLM_ZOO[model_name]["model_path"]
    processor = AutoProcessor.from_pretrained(model_path)
    if model_name == "llava_next":
        processor.patch_size = 14
    if model_name == "qwen2_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16)
    else:
        model = AutoModelForPreTraining.from_pretrained(model_path, torch_dtype=torch.float16)
    return model, processor

#%%

import torch
import torch.nn.functional as F

def call_model(model, processor, sample, device: str, output_type: str, REACTION_CLASSES: list, batch_size=len(REACTION_CLASSES)) -> dict:
    """
    Calls model and returns per-reaction log-probabilities (multi-token supported, batched for speed)
    """
    # 1. Format the input
    if output_type == "reaction_distribution":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an AI assistant. Your task is to forecast potential audience reactions to a movie scene. You will be provided with the video and some video context."},
                    {"type": "video"},
                    {"type": "text", "text": f"In this scene, {sample['clip_description']}."},
                    {"type": "text", "text": f"Given this clip, what do you think a viewer's reaction would be? Choose the letter of the most appropriate reaction:"},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Based on the scene, a viewer's reaction would be letter "}
                ]
            }
        ]

        # Create a mapping of options to reaction classes
        reaction_options = {}
        # Using letters A-U to accommodate all 21 reaction classes
        option_labels = [chr(65 + i) for i in range(21)]  # A-U in ASCII
        for i, reaction in enumerate(REACTION_CLASSES):
            reaction_options[option_labels[i]] = reaction
        # Get input_id of option_labels
        option_labels_input_ids = processor.tokenizer(option_labels, add_special_tokens=False)["input_ids"]
        option_labels_input_ids = [item[0] for item in option_labels_input_ids]
        
        # Add the options to the user's message
        options_text = ""
        for label, reaction in reaction_options.items():
            options_text += f"{label}. {reaction}\n"
        conversation[0]["content"].insert(-1, {"type": "text", "text": options_text})
    elif output_type == "reaction_dominant":
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an AI assistant. Your task is to forecast potential audience reactions to a movie scene. You will be provided with the video and some video context."},
                    {"type": "video"},
                    {"type": "text", "text": f"This is a movie scene from {sample['movie_name']}. In this scene, {sample['clip_description']}."},
                    {"type": "text", "text": f"What is the dominant audience reaction to this scene? Given this list of all reactions: {', '.join(REACTION_CLASSES)}, please provide the most likely reaction in one word."},
                ]
            }
        ]
    else:
        raise ValueError(f"Unknown output type: {output_type}")
    
    # Prepare the prompt (without any answer yet)
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    video_inputs = sample["video_input"]
    # Sample 16 frames only spacing them evenly
    num_frames = len(video_inputs)
    print(f"Number of frames: {num_frames}")
    if num_frames > 16:
        selected_frames = np.linspace(0, num_frames - 1, 16, dtype=int)
        video_inputs = [video_inputs[i] for i in selected_frames]

    # Also process the prompt alone (to find prompt length)
    inputs = processor(
        text=prompt,
        videos=video_inputs,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0,-1]
    # Get the logits for the reaction classe numbers in option_labels_input_ids
    logits = logits[option_labels_input_ids]
    # Get the softmax probabilities
    probs = torch.softmax(logits, dim=-1)
    return {
        "video_id": sample["video_id"],
        "reaction_probs": probs.cpu().numpy(),   # (num_reactions)
    }



#%%
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, accuracy_score

# Metrics for reaction_distribution (soft labels)
reaction_distribution_metrics = [
    "top_k_accuracy",  # Accurate if the target class is in top K predictions
    "precision_at_k",  # Precision@K
    "recall_at_k",  # Recall@K
    "macro_f1",  # Macro F1-score
    "micro_f1",  # Micro F1-score
    "weighted_f1",  # Weighted F1-score
    "mse",  # Mean Squared Error
    "js_divergence",  # Jensen-Shannon Divergence
    "pearson_correlation",  # Pearson correlation coefficient
    "spearman_correlation"  # Spearman rank correlation coefficient
]

# Metrics for reaction_dominant (single-label classification)
reaction_dominant_metrics = [
    "top_k_accuracy",  # Accurate if the target class is in top K predictions
    "precision_at_k",  # Precision@K
    "recall_at_k",  # Recall@K
    "macro_f1",  # Macro F1-score
    "micro_f1",  # Micro F1-score
    "weighted_f1"  # Weighted F1-score
]

evaluation_metrics = {
    "reaction_distribution": reaction_distribution_metrics,
    "reaction_dominant": reaction_dominant_metrics
}
#%%
from typing import Dict
def compute_metric(predicted_dict: Dict, target_dict: Dict, metric: str, k: int = 1) -> float:
    pass
    

#%%
from src.dataset import SENTIMENT_2_FINER_GRAINED_MAPPING, REACTION_CLASSES
from typing import List, Dict, Tuple
def map_emotion_to_sentiment(emotion: str) -> str:
    """
    Map a finer-grained emotion to its corresponding sentiment.
    """
    for sentiment, emotions in SENTIMENT_2_FINER_GRAINED_MAPPING.items():
        if emotion in emotions:
            return sentiment
    raise ValueError(f"Emotion '{emotion}' not found in SENTIMENT_2_FINER_GRAINED_MAPPING.")

def evaluate_model(generation_result_dict, dataset, output_type) -> Tuple[float, float]:
    """
    Evaluate the model
    """
    evaluation_results = {"top_k_accuracy@1" : None}
    n_accurate = 0
    predicted_outputs = []
    target_outputs = []
    num_unknown = 0
    for sample in dataset:
        video_id = sample["video_id"]
        if video_id not in generation_result_dict:
            continue
        
        predicted_output = generation_result_dict[video_id]["parsed_output"]
        target_output = sample[output_type]
        if predicted_output not in REACTION_CLASSES:
            print(f"Unknown predicted output: {predicted_output}")
            num_unknown += 1
            continue
        # Check if the predicted output is valid
        if output_type == "reaction_distribution":
            raise NotImplementedError("Reaction distribution evaluation is not implemented yet.")
        elif output_type == "reaction_dominant":
            predicted_outputs.append(predicted_output)
            target_outputs.append(target_output)
    evaluation_results["num_unknown"] = num_unknown
    evaluation_results["top_k_accuracy@1"] = accuracy_score(target_outputs, predicted_outputs)
    evaluation_results["precision_at_k@1"] = precision_score(target_outputs, predicted_outputs, average="macro")
    evaluation_results["recall_at_k@1"] = recall_score(target_outputs, predicted_outputs, average="macro")
    evaluation_results["macro_f1"] = f1_score(target_outputs, predicted_outputs, average="macro")
    evaluation_results["micro_f1"] = f1_score(target_outputs, predicted_outputs, average="micro")
    evaluation_results["weighted_f1"] = f1_score(target_outputs, predicted_outputs, average="weighted")

    # Compute performance with respect to sentiment of the dominant reaction
    predicted_sentiments = [map_emotion_to_sentiment(predicted_output) for predicted_output in predicted_outputs]
    target_sentiments = [map_emotion_to_sentiment(target_output) for target_output in target_outputs]
    evaluation_results["sentiment_top_k_accuracy@1"] = accuracy_score(target_sentiments, predicted_sentiments)
    evaluation_results["sentiment_precision_at_k@1"] = precision_score(target_sentiments, predicted_sentiments, average="macro")
    evaluation_results["sentiment_recall_at_k@1"] = recall_score(target_sentiments, predicted_sentiments, average="macro")
    evaluation_results["sentiment_macro_f1"] = f1_score(target_sentiments, predicted_sentiments, average="macro")
    evaluation_results["sentiment_micro_f1"] = f1_score(target_sentiments, predicted_sentiments, average="micro")
    evaluation_results["sentiment_weighted_f1"] = f1_score(target_sentiments, predicted_sentiments, average="weighted")
    return evaluation_results

#%%
from scipy.optimize import minimize_scalar
def rescale_prob(model_probs: np.array, reference_probs: np.array=None, alpha=None) -> np.array:

    def objective(temperature):
        total_tv = 0
        for i, model_prob_pair in enumerate(model_probs): 
            p=model_prob_pair ** (1.0/temperature) #scaled! 
            p=p/np.sum(p) # normalize! 
            q=reference_probs[i] 
            
            total_tv += 0.5*np.sum(np.abs(p-q))
        return total_tv
    if alpha is None:
        res = minimize_scalar(objective, bounds=(0.0, 10), method='bounded')
        print(res.x)
        alpha = res.x
    scaled = []
    for model_p in model_probs:
        scaled_p = model_p**(1.0/alpha)
        scaled.append(scaled_p/np.sum(scaled_p))
    scaled = np.array(scaled)

    return scaled, alpha
    
#%%
DEBUG = False
from IPython import get_ipython
if get_ipython() is not None:
    args = argparse.Namespace()
    args.metadata_dir = "/project/pi_mfiterau_umass_edu/trang/reaction-video-dataset/data/video2reaction-full"
    args.split = "test"
    args.key_frame_dir = "/project/pi_mfiterau_umass_edu/youtube_video/key_frames/"
    args.reaction_outcome_type = "reaction_distribution"
    args.temperature_scaling = True


    args.result_dir = "/scratch3/workspace/tramnguyen_umass_edu-email/reaction-video-dataset/results/video2reaction-full/vlm_zero_shot_multiple_choice/"
    args.model_name = "qwen2_vl"
    args.reprocess = False
    DEBUG = False

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

#%%
import json
import os
from src.metrics import compute_all_classification_metrics, compute_all_distribution_metrics

#%%
# python train_classification.py
def main():
    """
    Train a classification model
    """
    parser = argparse.ArgumentParser(description="Train a classification model")
    # to load data
    parser.add_argument("--metadata_dir", type=str, default="data/metadata", help="Directory containing the metadata files")
    parser.add_argument("--split", type=str, default="train", help="Split to use (train, val, test)")
    parser.add_argument("--key_frame_dir", type=str, default="data/processed_features", help="Directory containing the processed features")

    parser.add_argument("--reaction_outcome_type", type=str, default="reaction_distribution", help="Type of reaction outcome to predict")
    # to save model
    parser.add_argument("--model_name", type=str, default="llava_next", help="Name of the model to use")
    parser.add_argument("--result_dir", type=str, default="./results/", help="Directory to save the results")
    parser.add_argument("--reprocess", action="store_true", help="Reprocess the data")

    # temperature scaling
    parser.add_argument("--temperature_scaling", action="store_true", help="Use temperature scaling")
    args = parser.parse_args()
    #%%
    logging.basicConfig(level=logging.INFO)
    result_dir = os.path.join(args.result_dir, args.model_name)
    os.makedirs(result_dir, exist_ok=True)

    #%%
    if os.path.exists(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_targeted_distribution.npy")):
        targeted_distribution = np.load(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_targeted_distribution.npy"))
    else:
        targeted_distribution = None
    
    if os.path.exists(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution.npy")) and not args.reprocess:
        predicted_distribution = np.load(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution.npy"))
    else:
        predicted_distribution = None
    if os.path.exists(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution_scaled.npy")) and not args.reprocess:
        predicted_distribution_scaled = np.load(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution_scaled.npy"))
    else:
        predicted_distribution_scaled = None
    
    if targeted_distribution is not None and predicted_distribution is not None and not args.reprocess:
        logging.info(f"Already processed {args.split} set")
        evaluation_results = {}
        evaluation_results["classification"] = compute_all_classification_metrics(
            targeted_distribution,
            predicted_distribution,
        )
        evaluation_results["distribution"] = compute_all_distribution_metrics(
            targeted_distribution,
            predicted_distribution,
        )
        # Save
        evaluation_filepath = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_evaluation.json")
        with open(evaluation_filepath, "w") as f:
            json.dump(evaluation_results, f, indent=4, cls=NumpyTypeEncoder)
    if args.temperature_scaling and targeted_distribution is not None and predicted_distribution_scaled is not None and not args.reprocess:
        evaluation_results = {}
        evaluation_results["classification"] = compute_all_classification_metrics(
            targeted_distribution,
            predicted_distribution_scaled,
        )
        evaluation_results["distribution"] = compute_all_distribution_metrics(
            targeted_distribution,
            predicted_distribution_scaled,
        )
        # Save
        evaluation_filepath = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_evaluation_temperature_scaling.json")
        with open(evaluation_filepath, "w") as f:
            json.dump(evaluation_results, f, indent=4, cls=NumpyTypeEncoder)
        return

    # Otherwise, load the model and process the data
        #%%
    test_dataset = get_dataset(args.metadata_dir, args.key_frame_dir, args.split)
    #%%

    if args.temperature_scaling:
        #%%
        # Load the validation set
        # Check if validation result is already computed
        if os.path.exists(os.path.join(result_dir, f"fitted_temperature.npy")) and not args.reprocess:
            fitted_temperature = np.load(os.path.join(result_dir, f"fitted_temperature.npy"))
        else:
            val_prob_result = os.path.join(result_dir, f"{args.reaction_outcome_type}_val_generation.json")
            if os.path.exists(val_prob_result):
                with open(val_prob_result, "r") as f:
                    val_prob_result = json.load(f)
                val_dataset = get_dataset(args.metadata_dir, args.key_frame_dir, "val")
                val_predicted_distribution = []
                val_targeted_distribution = []
                for sample_i, sample in enumerate(val_dataset):
                    pred = np.array(val_prob_result[sample["video_id"]]["reaction_probs"])
                    val_predicted_distribution.append(pred)
                    sample_reaction = sample["reaction_distribution"]
                    # convert dict (reaction: distribution) to numpy array
                    target = np.zeros(len(REACTION_CLASSES))
                    for reaction, distribution in sample_reaction.items():
                        if reaction in REACTION_CLASSES:
                            target[REACTION_CLASSES.index(reaction)] = distribution
                    val_targeted_distribution.append(target)
                # Convert to numpy arrays
                val_predicted_distribution = np.array(val_predicted_distribution)
                val_targeted_distribution = np.array(val_targeted_distribution)
                # Compute the temperature scaling
                # Rescale the probabilities
                print("Fitting temperature scaling...")
                val_predicted_distribution, fitted_temperature = rescale_prob(val_predicted_distribution, val_targeted_distribution)
                # Save the fitted temperature
                np.save(os.path.join(result_dir, f"fitted_temperature.npy"), fitted_temperature)
            else:
                raise ValueError(f"Validation result not found at {val_prob_result}. Please run the validation first.")

    #%%
    torch.manual_seed(42)
    if not args.temperature_scaling:
        torch.set_grad_enabled(False)
        model, processor = get_model(args.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    
    #%%
    
    #%%
    # Generate results
    result = {}
    result_scaled = {}
    result_path = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_generation.json")
    result_scaled_path = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_generation_scaled.json")
    #%%
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            result = json.load(f)
    if args.temperature_scaling and os.path.exists(result_scaled_path):
        with open(result_scaled_path, "r") as f:
            result_scaled = json.load(f)
    
    
    if not args.temperature_scaling:
        predicted_distribution = []
        targeted_distribution = []
        for sample in tqdm(test_dataset):
            if sample["video_id"] in result and not args.reprocess:
                logging.info(f"Already processed {sample['video_id']}")
                pred = np.array(result[sample["video_id"]]["reaction_probs"])
            else:
                # Call the model
                try:
                    output = call_model(model, processor, sample, device, args.reaction_outcome_type, REACTION_CLASSES)
                    # Save the output
                    result[sample["video_id"]] = output
                    pred = output["reaction_probs"]
                    # Write the output to a file
                    with open(result_path, "w") as f:
                        json.dump(result, f, indent=4, cls=NumpyTypeEncoder)
                except Exception as e:
                    logging.error(f"Error processing {sample['video_id']}: {e}")
                    continue
            
            predicted_distribution.append(pred)
            sample_reaction = sample["reaction_distribution"]
            # convert dict (reaction: distribution) to numpy array
            target = np.zeros(len(REACTION_CLASSES))
            for reaction, distribution in sample_reaction.items():
                if reaction in REACTION_CLASSES:
                    target[REACTION_CLASSES.index(reaction)] = distribution
            targeted_distribution.append(target)
        

        predicted_distribution = np.array(predicted_distribution)
        targeted_distribution = np.array(targeted_distribution)
    
        #%%
        # Save the targeted distribution
        np.save(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_targeted_distribution.npy"), targeted_distribution)
        #%%
        # Save the predicted distribution
        np.save(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution.npy"), predicted_distribution)
        
    #%%
    if args.temperature_scaling:
        predicted_distribution = np.load(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution.npy"))
        # Load the fitted temperature
        fitted_temperature = np.load(os.path.join(result_dir, f"fitted_temperature.npy"))
        # Rescale the probabilities
        predicted_distribution, fitted_temperature = rescale_prob(predicted_distribution, alpha=fitted_temperature)
        # Save the scaled results
        for i, sample in enumerate(test_dataset):
            video_id = sample["video_id"]
            result_scaled[video_id] = {
                "video_id": video_id,
                "reaction_probs": predicted_distribution[i],
            }
        # Write the scaled output to a file
        result_scaled_path = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_generation_scaled.json")
        with open(result_scaled_path, "w") as f:
            json.dump(result_scaled, f, indent=4, cls=NumpyTypeEncoder)
        np.save(os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_predicted_distribution_scaled.npy"), predicted_distribution)
   
    #%%
    # Evaluate the model
    evaluation_results = {}
    evaluation_results["classification"] = compute_all_classification_metrics(
        np.array(targeted_distribution),
        np.array(predicted_distribution),
    )
    evaluation_results["distribution"] = compute_all_distribution_metrics(
        np.array(targeted_distribution),
        np.array(predicted_distribution),
    )
    #%%
    # Save the results
    if args.temperature_scaling:
        evaluation_filepath = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_evaluation_temperature_scaling.json")
    else:
        evaluation_filepath = os.path.join(result_dir, f"{args.reaction_outcome_type}_{args.split}_evaluation.json")
    with open(evaluation_filepath, "w") as f:
        json.dump(evaluation_results, f, indent=4, cls=NumpyTypeEncoder)

    
    
# %%
# python train_classification.py 
if __name__ == "__main__":
    main()
