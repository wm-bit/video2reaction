"""
This script uses a pretrained LLM to generate finer-grained reaction for comment
"""
import torch

ekman_mapping = {
    "anger": ["anger", "annoyance", "disapproval"],
    "disgust": ["disgust"],
    "fear": ["fear", "nervousness"],
    "joy": ["joy", "amusement", "approval", "excitement", "gratitude",  "love", "optimism", "relief", "pride", "admiration", "desire", "caring"],
    "sadness": ["sadness", "disappointment", "embarrassment", "grief",  "remorse"],
    "surprise": ["surprise", "realization", "confusion", "curiosity"],
    "neutral": ["neutral"]
}

def map_fine_grained_to_high_level_reaction(fine_grained_reaction):
    high_level_reaction = []
    for reaction in fine_grained_reaction:
        for ekman_reaction, ekman_emotions in ekman_mapping.items():
            if reaction in ekman_emotions:
                high_level_reaction.append(ekman_reaction)
    return high_level_reaction


def predict_sentiment(model, comment, tokenizer):

def analyze_comment(model, comment_dict, tokenizer, top_k=3, prob_threshold=0.2):
    # TODO: Implement this function
    result = {}
    for comment_id, comment in comment_dict.items():
        # Tokenize the comment
        if "rephrased_comment" in comment.keys():
            if comment["rephrased_comment"] is None or comment["rephrased_comment"] == "None" or comment["rephrased_comment"] is False:
                results[comment_id] = {
                    "comment": comment["comment"],
                    "rephrased_comment": None,
                    "parsed_output": None
                }
                continue
            
            inputs = tokenizer(comment["rephrased_comment"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        else
            inputs = tokenizer(comment["comment"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        # Predict the label
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
            probs, indices = torch.topk(probs, top_k)
            labels = []
            for prob, idx in zip(probs, indices):
                if prob > prob_threshold:
                    label = model.config.id2label[idx.item()]
                    labels.append(label)
        
        high_level_reaction = map_fine_grained_to_high_level_reaction(labels)
        result[comment_id] = {
            "comment": comment["comment"],
            "rephrased_comment": comment.get("rephrased_comment", None),
            "parsed_output": {
                "high_level_reaction": high_level_reaction,
                "finer_grained_reaction": labels
            }
        }
    return result