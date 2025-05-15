"""
Script containing the dataset classes for the project.
"""
#%%
from collections import defaultdict
import json
import os
from typing import Dict, List
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torchaudio
import pickle
#%%

# Define the fixed class order
REACTION_CLASSES = ['sadness',
 'disgust',
 'grief',
 'fear',
 'disapproval',
 'disappointment',
 'embarrassment',
 'nervousness',
 'annoyance',
 'anger',
 'confusion',
 'realization',
 'caring',
 'curiosity',
 'relief',
 'approval',
 'surprise',
 'excitement',
 'amusement',
 'admiration',
 'joy']

VAD_LEXICON = json.load(open("./data/nrc-python/nrc_vad.json"))
REACTION_CLASSES_VAD = {reaction: VAD_LEXICON[reaction] for reaction in REACTION_CLASSES}
REACTION_CLASSES_SORTED_BY_VA = sorted(REACTION_CLASSES, key=lambda x: (REACTION_CLASSES_VAD[x]["valence"], REACTION_CLASSES_VAD[x]["arousal"]))
REACTION_CLASSES = REACTION_CLASSES_SORTED_BY_VA
REACTION_VA_MATRIX = np.zeros((len(REACTION_CLASSES), 2))
for i, reaction in enumerate(REACTION_CLASSES):
    if reaction in REACTION_CLASSES_VAD:
        REACTION_VA_MATRIX[i, 0] = REACTION_CLASSES_VAD[reaction]["valence"]
        REACTION_VA_MATRIX[i, 1] = REACTION_CLASSES_VAD[reaction]["arousal"]
    else:
        print(f"Warning: {reaction} not found in VAD lexicon.")


ALL_GENRES = ['Music', 'Family', 'Crime', 'Thriller', 'Action', 'Western', 'Sci-Fi', 'Short', 'History', 'Adventure', 'Fantasy', 'Romance', 'Film-Noir', 'Biography', 'Comedy', 'Musical', 'War', 'Horror', 'Animation', 'Documentary', 'Sport', 'Mystery', 'Drama']
#%%

NUM_CLASSES = len(REACTION_CLASSES)
REACTION_INDEX = {reaction: idx for idx, reaction in enumerate(REACTION_CLASSES)}
SENTIMENT_2_FINER_GRAINED_MAPPING = {
"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
"negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
"ambiguous": ["realization", "surprise", "curiosity", "confusion"],
"neutral": ["neutral"]
}
SENTIMENT_CLASSES = ["positive", "negative", "ambiguous"] # removed neutral
SENTIMENT_2_FINER_GRAINED_INDICES_MAPPING = {
    "positive": [REACTION_INDEX[reaction] for reaction in SENTIMENT_2_FINER_GRAINED_MAPPING["positive"] if reaction in REACTION_INDEX],
    "negative": [REACTION_INDEX[reaction] for reaction in SENTIMENT_2_FINER_GRAINED_MAPPING["negative"] if reaction in REACTION_INDEX],
    "ambiguous": [REACTION_INDEX[reaction] for reaction in SENTIMENT_2_FINER_GRAINED_MAPPING["ambiguous"] if reaction in REACTION_INDEX],
}



def load_key_frames(key_frame_dir) -> np.ndarray:
    try:
        key_frame_list = pd.read_csv(f"{key_frame_dir}/index.csv")
    except pd.errors.EmptyDataError:
        print(f"Empty key frame index file {key_frame_dir}/index.csv")
        return False
    scene_list = key_frame_list["scene_number"].values
    frames = []
    for scene_number in scene_list:
        # if scene number is 1 then file name is 001.jpg
        scene_number_str = str(scene_number + 1).zfill(3)
        key_frame_path = f"{key_frame_dir}/{scene_number_str}.jpg"
        if not os.path.exists(key_frame_path):
            print(f"Missing key frame images {key_frame_path}")
            return False
        frames.append(Image.open(key_frame_path))
    # Convert to numpy array
    # frames = np.stack([x.to_ndarray(format="rgb24") for x in frames])
    return frames

def load_audio_wav(raw_dir, video_id, sampling_rate=16000):
    audio_path = f"{raw_dir}/{video_id}/audio.wav"
    waveform, sr = torchaudio.load(audio_path)
    if sr != sampling_rate:
        waveform = torchaudio.transforms.Resample(sr, sampling_rate)(waveform)
    return waveform.squeeze(0)

def pad_or_sample_frames(frames, target_frames=32, do_pad=True):
    """
    Adjusts the number of frames to match the required input size for ViViT.

    Args:
        frames (list of PIL.Image or numpy arrays): List of extracted frames.
        target_frames (int): Number of frames required (default: 32).

    Returns:
        list: A list of 32 frames.
    """
    num_frames = len(frames)
    
    if num_frames < target_frames and do_pad:
        # Pad by repeating the last frame
        frames += [frames[-1]] * (target_frames - num_frames)
    elif num_frames > target_frames:
        # Sample 32 evenly spaced frames
        indices = np.linspace(0, num_frames - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    return frames

#%%
def map_sequence_to_half_frames(seq_feats: torch.Tensor, num_frames: int = 32, tubelet_t: int = 2) -> torch.Tensor:
    """
    Maps a [sequence_len, 768] ViViT output to a [num_frames // 2, 768] representation
    by assigning one feature per tubelet temporal chunk (removing duplicate frames).

    Args:
        seq_feats (torch.Tensor): Shape [3137, 768] including CLS token.
        num_frames (int): Number of frames in the video (typically 32).
        tubelet_t (int): Temporal size of the tubelet (typically 2).

    Returns:
        torch.Tensor: Shape [num_frames // 2, 768]
    """
    assert seq_feats.shape[0] == 3137, "Expected sequence length to be 3137 (including CLS token)."
    feat_dim = seq_feats.shape[1]
    
    # Drop CLS token
    tubelet_feats = seq_feats[1:]  # [3136, 768]

    # Tubelet layout
    num_t, num_h, num_w = 16, 14, 14  # from your config
    assert num_t * tubelet_t == num_frames, "Mismatch between tubelet_t, num_t, and num_frames."

    # Now, instead of mapping every frame, map each tubelet time block (of 2 frames) once
    frame_feats = []
    for t_idx in range(num_t):
        # Collect all tubelets at this time index (across space)
        tubelet_indices = [
            t_idx * (num_h * num_w) + idx for idx in range(num_h * num_w)
        ]
        frame_feat = tubelet_feats[tubelet_indices].mean(dim=0)  # average over spatial tubelets
        frame_feats.append(frame_feat)

    return torch.stack(frame_feats)  # [num_frames // 2, 768]

#%%
class Video2Reaction(Dataset):
    def __init__(self, metadata_file_path, 
            processed_feature_dir=None, 
            visual_encoder="vivit", 
            text_encoder="bert-base-uncased", 
            audio_encoder_acoustic="clap_general",
            audio_encoder_semantic="hubert_large",
            use_time_dimension=False,
            device="cuda",
            lazy_load=True,
            input_type="processed", # processed / raw / vlm
            key_frame_dir=None,
            comment_level=False,
            comment_level_metadata_file_path=None,
            max_sample_size=None,
            cache_file_path=None,
            ):
        """
        Args:
            metadata_file_path (str): Path to the JSON metadata file.
            processed_feature_dir (str): Directory where processed features are stored.
            visual_encoder (str): Type of visual encoder (used for feature path).
            text_encoder (str): Name of the Hugging Face BERT model.
            audio_encoder_acoustic (str): Acoustic audio encoder (e.g., clap_general).
            audio_encoder_semantic (str): Semantic audio encoder (e.g., hubert_large).
            device (str): "cpu" or "cuda" for model inference.
        """
        self.metadata = json.load(open(metadata_file_path))
        if max_sample_size is not None:
            self.metadata = dict(list(self.metadata.items())[:max_sample_size])
        self.comment_level = comment_level
        self.key_frame_dir = key_frame_dir
        self.use_time_dimension = use_time_dimension
        self.audio_encoder_acoustic = audio_encoder_acoustic
        self.audio_encoder_semantic = audio_encoder_semantic
        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder
        self.device = device

        # Remove videos with missing features
        video_ids = list(self.metadata.keys())
        missing_videos = []

        # if VLM input
        self.input_type = input_type
        self.comment_level_metadata = json.load(open(comment_level_metadata_file_path)) if comment_level else None
        self.preloaded_video_input = {}
        if self.input_type == "vlm" and not lazy_load:
            if cache_file_path is not None and os.path.exists(cache_file_path):
                print(f"Loading preloaded video input from {cache_file_path}")
                self.preloaded_video_input = pickle.load(open(cache_file_path, "rb"))
            else:
                for video_id in tqdm(video_ids):
                    video_input = load_key_frames(os.path.join(key_frame_dir, video_id))
                    if video_input is False:
                        missing_videos.append(video_id)
                        del self.metadata[video_id]
                    else:
                        self.preloaded_video_input[video_id] = pad_or_sample_frames(video_input, target_frames=16, do_pad=False)
                # Save the preloaded video input to cache
                if cache_file_path is not None:
                    print(f"Saving preloaded video input to {cache_file_path}")
                    pickle.dump(self.preloaded_video_input, open(cache_file_path, "wb"))
        
        if self.input_type == "processed":
            self.feature_filename = "full.npy" if use_time_dimension else "pooler_mean.npy"
            # for video_id in video_ids:
            #     visual_feature_path = os.path.join(processed_feature_dir, "visual", visual_encoder, video_id, self.feature_filename)
            #     audio_acoustic_feature_path = os.path.join(processed_feature_dir, "audio", audio_encoder_acoustic, video_id, self.feature_filename)
            #     audio_semantic_feature_path = os.path.join(processed_feature_dir, "audio", audio_encoder_semantic, video_id, self.feature_filename)

            #     if not os.path.exists(visual_feature_path) or not os.path.exists(audio_acoustic_feature_path) or not os.path.exists(audio_semantic_feature_path):
            #         missing_videos.append(video_id)
            #         del self.metadata[video_id]
            self.missing_videos = missing_videos
            print(f"Removed {len(missing_videos)} videos with missing features.")

        self.video_ids = list(self.metadata.keys())
        self.processed_feature_dir = processed_feature_dir
        if self.input_type == "processed":
            self.tokenizer = AutoTokenizer.from_pretrained(text_encoder)
            self.bert_model = AutoModel.from_pretrained(text_encoder).to(device).eval()  # Load BERT in eval mode

        # Lazy loading flag
        self.lazy_load = lazy_load
        self.preloaded_data = {}
        if self.input_type == "processed":
            if not lazy_load:
                if cache_file_path is not None and os.path.exists(cache_file_path):
                    print(f"Loading preloaded embeddings from {cache_file_path}")
                    self.preloaded_data = torch.load(cache_file_path, map_location=device)
                else:
                    for video_id in tqdm(self.video_ids):
                        clip_description = self.metadata[video_id]["clip_description"]
                        self.preloaded_data[video_id] = {
                            "clip_description_embedding": self._get_bert_embedding(clip_description),
                            "visual_feature": self._get_visual_features(video_id),
                            "audio_acoustic_feature": self._get_audio_features(video_id, self.audio_encoder_acoustic),
                            "audio_semantic_feature": self._get_audio_features(video_id, self.audio_encoder_semantic)
                        }
                    if cache_file_path is not None:
                        print(f"Saving preloaded embeddings to {cache_file_path}")
                        torch.save(self.preloaded_data, cache_file_path)
        elif self.input_type == "raw":
            if not lazy_load:
                if cache_file_path is not None and os.path.exists(cache_file_path):
                    print(f"Loading preloaded data from {cache_file_path}")
                    self.preloaded_data = torch.load(cache_file_path)
                else:
                    for video_id in tqdm(self.video_ids):
                        self.preloaded_data[video_id] = {
                            "clip_description": self.metadata[video_id]["clip_description"],
                            "visual_feature": load_key_frames(os.path.join(key_frame_dir, video_id)),
                            "audio_feature": load_audio_wav(video_id),
                        }
                    if cache_file_path is not None:
                        print(f"Saving preloaded data to {cache_file_path}")
                        torch.save(self.preloaded_data, cache_file_path)
            


    def _get_audio_features(self, video_id, audio_encoder):
        """Loads audio features for the given video ID and encoder."""
        audio_feature_path = os.path.join(self.processed_feature_dir, "audio", audio_encoder, video_id, self.feature_filename)
        if os.path.exists(audio_feature_path):
            return torch.tensor(np.load(audio_feature_path), dtype=torch.float32).squeeze(0)
        else:
            raise FileNotFoundError(f"Audio feature not found for video ID: {video_id} with encoder: {audio_encoder}")

    def _get_visual_features(self, video_id):
        """Loads visual features for the given video ID."""
        visual_feature_path = os.path.join(self.processed_feature_dir, "visual", self.visual_encoder, video_id, self.feature_filename)
        if os.path.exists(visual_feature_path):
            visual_features = torch.tensor(np.load(visual_feature_path), dtype=torch.float32).squeeze(0)
            # map back to frame
            if self.use_time_dimension:
                if self.visual_encoder == "vivit":
                    visual_features = map_sequence_to_half_frames(visual_features, num_frames=32, tubelet_t=2)
            return visual_features

        else:
            raise FileNotFoundError(f"Visual feature not found for video ID: {video_id}")
    
    def _get_bert_embedding(self, text):
        """Generates BERT embedding for the given text."""
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        return outputs.pooler_output.squeeze(0)  # Get the pooled output (CLS token representation)
    def _compute_genre_encoding(self, genre):
        """
        Computes one-hot encoding for the genre.
        """
        genre_encoding = torch.zeros(len(ALL_GENRES))
        if genre is None:
            return genre_encoding
        # If genre is a single string, convert it to a list
        if isinstance(genre, str):
            genre = [genre]

        for g in genre:
            if g in ALL_GENRES:
                genre_encoding[ALL_GENRES.index(g)] = 1.0
        
        return genre_encoding
    
    def __len__(self):
        """Returns the number of videos in the dataset."""
        return len(self.video_ids)

    
    def __getitem__(self, idx):
        """
        Returns:
        - video_id (str)
        - BERT sentence embedding (tensor)
        - visual features (tensor)
        - audio acoustic features (tensor)
        - audio semantic features (tensor)
        - reaction_dominant (str) [to be one-hot encoded in collate_fn]
        - reaction_distribution (dict) [to be mapped in collate_fn]
        """
        video_id = self.video_ids[idx]
        video_meta = self.metadata[video_id]

        if self.input_type == "vlm":
            if not self.comment_level:
                # If lazy loading is enabled, load the features on demand
                if self.lazy_load:
                    video_input = load_key_frames(os.path.join(self.key_frame_dir, video_id))
                    if video_input is False:
                        raise FileNotFoundError(f"Key frame not found for video ID: {video_id}")
                    else:
                        video_input = pad_or_sample_frames(video_input, target_frames=16, do_pad=False)
                else:
                    video_input = self.preloaded_video_input[video_id]
                return {
                    "video_id": video_id,
                    "video_input": video_input,
                    "movie_name": video_meta["movie_name"],
                    "movie_genre": video_meta["genre"],
                    "clip_description": video_meta["clip_description"],
                    "reaction_dominant": video_meta["reaction_outcome"]["dominant_reaction"],
                    "reaction_distribution": video_meta["reaction_outcome"]["reaction_distribution"],
                }

        sentiment_distribution = {}
        for sentiment, reactions in SENTIMENT_2_FINER_GRAINED_MAPPING.items():
            sentiment_distribution[sentiment] = sum([video_meta["reaction_outcome"]["reaction_distribution"].get(reaction, 0) for reaction in reactions])

        # Convert reaction_dominant to integer
        # Convert reaction_distribution to an array (NUM_CLASSES,)
        reaction_distribution = torch.zeros((NUM_CLASSES,))
        for reaction, value in video_meta["reaction_outcome"]["reaction_distribution"].items():
            if reaction in REACTION_INDEX:
                reaction_distribution[REACTION_INDEX[reaction]] = value

        if self.input_type == "processed":
            # If lazy loading is enabled, load the features on demand
            if self.lazy_load:
                text_embedding = self._get_bert_embedding(video_meta["clip_description"])
                visual_features = self._get_visual_features(video_id)
                audio_acoustic_features = self._get_audio_features(video_id, self.audio_encoder_acoustic)
                audio_semantic_features = self._get_audio_features(video_id, self.audio_encoder_semantic)
            else:
                text_embedding = self.preloaded_data[video_id]["clip_description_embedding"]
                visual_features = self.preloaded_data[video_id]["visual_feature"]
                audio_acoustic_features = self.preloaded_data[video_id]["audio_acoustic_feature"]
                audio_semantic_features = self.preloaded_data[video_id]["audio_semantic_feature"]

            
            return {
                "video_id": video_id,
                "movie_genre": self._compute_genre_encoding(video_meta["genre"]),
                "clip_description_embedding": text_embedding,
                "visual_feature": visual_features,
                "audio_acoustic_feature": audio_acoustic_features,
                "audio_semantic_feature": audio_semantic_features,
                "reaction_dominant": video_meta["reaction_outcome"]["dominant_reaction"],
                "reaction_distribution": reaction_distribution,
                
            }

        if self.input_type == "raw":
            if self.lazy_load:
                return {
                    "video_id": video_id,
                    "movie_genre": video_meta["genre"],
                    "clip_description": video_meta["clip_description"],
                    "visual_feature": load_key_frames(os.path.join(self.key_frame_dir, video_id)),
                    "audio_feature": load_audio_wav(video_id),
                    "reaction_dominant": video_meta["reaction_outcome"]["dominant_reaction"],
                    "reaction_distribution": reaction_distribution,
                    
                }
            else:
                return {
                    "video_id": video_id,
                    "movie_genre": video_meta["genre"],
                    "clip_description": video_meta["clip_description"],
                    "visual_feature": self.preloaded_data[video_id]["visual_feature"],
                    "audio_feature": self.preloaded_data[video_id]["audio_feature"],
                    "reaction_dominant": video_meta["reaction_outcome"]["dominant_reaction"],
                    "reaction_distribution": reaction_distribution,
                    
                }
# %%

#%%
def collate_fn(batch):
    """
    Custom collate function for batch processing.

    Args:
        batch (list of dicts): List of dataset items.

    Returns:
        dict: Batched tensors with padded BERT embeddings, visual features,
              one-hot encoded `reaction_dominant`, and formatted `reaction_distribution`.
    """
    batch_size = len(batch)

    # Extract raw data
    video_ids = [item["video_id"] for item in batch]
    text_embeddings = [item["clip_description_embedding"] for item in batch]
    visual_features = [item["visual_feature"] for item in batch]
    audio_acoustic_features = [item["audio_acoustic_feature"] for item in batch]
    audio_semantic_features = [item["audio_semantic_feature"] for item in batch]

    # Create tensors for text embeddings
    text_embeddings = torch.stack(text_embeddings)

    # Pad visual features, audio_acoustic_features, and audio_semantic_features along the time dimension
    def pad_features(features):
        assert len(features[0].shape) == 2, "Features should be 2D tensors"
        #print(f"Padding features: {features[0].shape}")
        max_time = max(f.shape[0] for f in features)  # Find the maximum time dimension
        feature_dim = features[0].shape[1]  # Feature dimension (D)
        padded_features = torch.zeros((batch_size, max_time, feature_dim), dtype=torch.float32)
        #print(padded_features.shape)
        for i, f in enumerate(features):
            #print(f.shape)
            padded_features[i, :f.shape[0], :] = f  # Copy the feature values
        return padded_features


    if len(visual_features[0].shape) == 2:
        visual_features = pad_features(visual_features)
    else:
        visual_features = torch.stack(visual_features) # B, D
    if len(audio_acoustic_features[0].shape) == 2:
        audio_acoustic_features = pad_features(audio_acoustic_features)
    else:
        audio_acoustic_features = torch.stack(audio_acoustic_features)
    if len(audio_semantic_features[0].shape) == 2:
        audio_semantic_features = pad_features(audio_semantic_features)
    else:
        audio_semantic_features = torch.stack(audio_semantic_features)

    # One-hot encode dominant reaction and format distribution
    reaction_dominant_tensor = torch.zeros((batch_size, NUM_CLASSES), dtype=torch.float32)
    reaction_distribution_tensor = torch.stack([item["reaction_distribution"] for item in batch])
    #sentiment_distribution_tensor = torch.stack([item["sentiment_distribution"] for item in batch])
    for i, item in enumerate(batch):
        dominant_reaction = item["reaction_dominant"]
        if dominant_reaction in REACTION_INDEX:
            reaction_dominant_tensor[i, REACTION_INDEX[dominant_reaction]] = 1.0


    return {
        "video_id": video_ids,
        "input": {
            "clip_description_embedding": text_embeddings,
            "visual_feature": visual_features,
            "audio_acoustic_feature": audio_acoustic_features,
            "audio_semantic_feature": audio_semantic_features,
            "movie_genre": torch.stack([item["movie_genre"] for item in batch]),
        },
        "output": {
            "reaction_dominant": reaction_dominant_tensor,
            "reaction_distribution": reaction_distribution_tensor,
            #"sentiment_distribution": sentiment_distribution_tensor,
        },
    }


#%%
def reaction_dist_to_dict(reaction_distribution):
    """
    Converts the reaction distribution to a dictionary format.
    """
    reaction_distribution_dict = {}
    for reaction in REACTION_CLASSES:
        reaction_distribution_dict[reaction] = reaction_distribution[REACTION_INDEX[reaction]]
    return reaction_distribution_dict

def print_sorted_reaction_distribution(reaction_distribution, k=10):
    """
    Prints the reaction distribution in a sorted manner.
    """
    # Convert to dictionary
    reaction_distribution_dict = reaction_dist_to_dict(reaction_distribution)
    # Sort by value
    sorted_reactions = sorted(reaction_distribution_dict.items(), key=lambda x: x[1], reverse=True)
    for reaction, value in sorted_reactions[:k]:
        print(f"{reaction}: {value:.4f}")


