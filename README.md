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

## About the dataset

* Full dataset access:  https://huggingface.co/datasets/video2reac/Video2Reaction/

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

## Running instructions
* To create virtual environment
```bash
python -m venv reaction-video-venv
pip install -r reaction-video-venv-requirements.txt
```
* To run LDL baseline
```bash
python run_ldl_baselines.py \
    --metadata_dir "/project/metadata/dir" \
    --cache_folder "/project/cache_folder" 
    --visual_encoder "vit" \
    --text_encoder "bert-base-uncased" \
    --audio_encoder_acoustic "clap_general" \
    --audio_encoder_semantic "hubert_large" \
    --batch_size 500 \
    --epochs 200 \
    --checkpoint_base_dir "/project/checkpoint_dir" \
    --methods SA_BFGS \
    --seed 320
```
* To run CTEN baseline
in `src-mmim-cten` folder:
```bash
python scripts/train_cten.py --config vaanet.yaml --seed_idx 0 --audio_mode average --is_erasing 0 --loss CrossEntropyLoss
```
Training configs/hyperparams are included in `src-mmin-cten/config/v2r/vaanet.yaml`

* To run MMIM baseline
in `src-mmim-cten` folder:
```bash
python scripts/train_mminfomax.py --config mminfomax.yaml --seed_idx 0 --audio_mode average --loss CrossEntropyLoss
```
Training configs/hyperparams are included in `src-mmin-cten/config/v2r/mminfomax.yaml`
