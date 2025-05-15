# Video2Reaction Benchmark

## To load the dataset
```{python}
from src.dataset import Video2Reaction
visual_encoder = "vit"
audio_encoder_acoustic="clap_general"
audio_encoder_semantic="hubert_large"
text_encoder = "bert-base-uncased"
# Cache file can be downloaded from https://huggingface.co/datasets/video2reac/Video2Reaction/tree/main
split = "train"
train_dataset = Video2Reaction(metadata_file_path=os.path.join(metadata_dir, f"{split}.json"), 
                                 processed_feature_dir=processed_feature_dir, 
                                 visual_encoder=visual_encoder, 
                                 text_encoder=text_encoder, 
                                 audio_encoder_acoustic=audio_encoder_acoustic, 
                                 audio_encoder_semantic=audio_encoder_semantic,
                                 lazy_load=False, use_time_dimension=True,
                                 cache_file_path=f"{cache_folder}/{split}_{visual_encoder}_{text_encoder}_{audio_encoder_acoustic}_{audio_encoder_semantic}.pt")
```

# About the dataset

* REACTION_CLASSES
```python
REACTION_CLASSES = ['sadness', 'disgust', 'grief', 'fear', 'disapproval', 'disappointment', 'embarrassment', 'nervousness', 'annoyance', 'anger', 'confusion', 'realization', 'caring', 'curiosity', 'relief', 'approval', 'surprise', 'excitement', 'amusement', 'admiration', 'joy']
```
* MOVIW_GENRES
```python
MOVIW_GENRES = ['Music', 'Family', 'Crime', 'Thriller', 'Action', 'Western', 'Sci-Fi', 'Short', 'History', 'Adventure', 'Fantasy', 'Romance', 'Film-Noir', 'Biography', 'Comedy', 'Musical', 'War', 'Horror', 'Animation', 'Documentary', 'Sport', 'Mystery', 'Drama']
```
* SENTIMENT_2_FINER_GRAINED_MAPPING
```python
SENTIMENT_2_FINER_GRAINED_MAPPING = {
"positive": ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"],
"negative": ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"],
"ambiguous": ["realization", "surprise", "curiosity", "confusion"]
}
```
