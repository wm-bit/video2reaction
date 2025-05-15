"""
Define a pytorch class to train a classification model
"""
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
from src.tcn import TemporalConvNet


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
    def forward(self, input, context):
        # input: (B, T, D)
        # context: (B, D)
        input = input.permute(1, 0, 2)  # (T, B, D)
        context = context.unsqueeze(0).repeat(input.size(0), 1, 1)  # (T, B, D)
        output, _ = self.attention(input, context, context)
        return output.permute(1, 0, 2)  # (B, T, D)

class FramewiseContextAttention(nn.Module):
    def __init__(self, feat_dim, context_dim, hidden_dim=128):
        super(FramewiseContextAttention, self).__init__()
        
        # MLP to produce a score for each frame given audio+context
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single score per frame
        )

    def forward(self, features, context_features):
        """
        Args:
            features: Tensor of shape (batch_size, num_frames, audio_dim)
            context_features: Tensor of shape (batch_size, context_dim)
        Returns:
            attended_feat: Tensor of shape (batch_size, audio_dim)
        """
        batch_size, num_frames, audio_dim = features.shape

        # Expand context to match each frame
        context_expanded = context_features.unsqueeze(1).repeat(1, num_frames, 1)  # (batch_size, num_frames, context_dim)

        # Concatenate audio feature with context at each frame
        concat = torch.cat([features, context_expanded], dim=-1)  # (batch_size, num_frames, audio_dim + context_dim)
        #print(f"Concat shape: {concat.shape}")

        # Pass through MLP to get attention scores
        scores = self.mlp(concat).squeeze(-1)  # (batch_size, num_frames)

        # Normalize scores to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch_size, num_frames)

        # Weighted sum of audio features
        attended_audio = torch.bmm(attn_weights.unsqueeze(1), features).squeeze(1)  # (batch_size, audio_dim)

        return attended_audio

class MultimodalReactionPredictor(nn.Module):
    def __init__(self, embedding_dims, num_genre, num_reaction_classes, tcn_channel, tcn_attention, kernel_size, modalities, context_attention_hidden_dim):
        """
        modalities: dict controlling which modalities are active, e.g.,
            {
                'audio_acoustic': True,
                'audio_semantic': True,
                'visual': True,
                'clip_description': True,
                'genre': True
            }
        """
        super(MultimodalReactionPredictor, self).__init__()
        self.modalities = modalities

        # Temporal Conv Nets
        if self.modalities.get('audio_acoustic', False):
            self.audio_acoustic_tcn = TemporalConvNet(num_inputs=embedding_dims['audio_acoustic'], 
                                                      max_length=300, 
                                                      num_channels=tcn_channel['audio_acoustic'], 
                                                      attention=tcn_attention, 
                                                      kernel_size=kernel_size, dropout=0.1)
        
        if self.modalities.get('audio_semantic', False):
            self.audio_semantic_tcn = TemporalConvNet(num_inputs=embedding_dims['audio_semantic'], 
                                                      max_length=100, 
                                                      num_channels=tcn_channel['audio_semantic'], 
                                                      attention=tcn_attention, 
                                                      kernel_size=kernel_size, dropout=0.1)
        
        if self.modalities.get('visual', False):
            self.visual_tcn = TemporalConvNet(num_inputs=embedding_dims['visual'], 
                                              max_length=100, 
                                              num_channels=tcn_channel['visual'], 
                                              attention=tcn_attention, 
                                              kernel_size=kernel_size, dropout=0.1)

        # Dimensions after TCN
        self.tcn_dims = {}
        if self.modalities.get('audio_acoustic', False):
            self.tcn_dims['audio_acoustic'] = tcn_channel['audio_acoustic'][-1]
        if self.modalities.get('audio_semantic', False):
            self.tcn_dims['audio_semantic'] = tcn_channel['audio_semantic'][-1]
        if self.modalities.get('visual', False):
            self.tcn_dims['visual'] = tcn_channel['visual'][-1]

        # Cross-attention
        context_dim = 0
        if self.modalities.get('clip_description', False):
            context_dim += embedding_dims['clip_description']
        if self.modalities.get('genre', False):
            context_dim += num_genre
        print(f"Context dimension: {context_dim}")
        self.context_proj = nn.Linear(context_dim, self.tcn_dims['audio_acoustic'])  # Project context to match TCN output dimensions

        if self.modalities.get('audio_acoustic', False):
            self.cross_attention_audio_acoustic = FramewiseContextAttention(feat_dim=self.tcn_dims['audio_acoustic'], context_dim=self.tcn_dims['audio_acoustic'], hidden_dim=context_attention_hidden_dim)
        if self.modalities.get('audio_semantic', False):
            self.cross_attention_audio_semantic = FramewiseContextAttention(feat_dim=self.tcn_dims['audio_semantic'], context_dim=self.tcn_dims['audio_acoustic'], hidden_dim=context_attention_hidden_dim)
        if self.modalities.get('visual', False):
            self.cross_attention_visual = FramewiseContextAttention(feat_dim=self.tcn_dims['visual'], context_dim=self.tcn_dims['audio_acoustic'], hidden_dim=context_attention_hidden_dim)

        # Final MLP
        total_dim = sum(self.tcn_dims.values())
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_reaction_classes)
        )

    def build_context(self, clip_description_emb, genre):
        context = []
        if self.modalities.get('clip_description', False):
            context.append(clip_description_emb)
        if self.modalities.get('genre', False):
            context.append(genre)
        if context:
            return torch.cat(context, dim=-1)  # (B, total_context_dim)
        else:
            raise ValueError("At least one context (clip_description or genre) must be enabled.")

    def forward(self, audio_acoustic_emb, audio_semantic_emb, visual_emb, clip_description_emb, genre):
        features = []

        # Build cross-attention context
        context = self.build_context(clip_description_emb, genre)
        # Project context to match TCN output dimensions
        context = self.context_proj(context)

        if self.modalities.get('audio_acoustic', False):
            # Reshape to (B, D, T)
            audio_acoustic_emb = audio_acoustic_emb.permute(0, 2, 1)  # (B, D, T)
            audio_acoustic_repr = self.audio_acoustic_tcn(audio_acoustic_emb)

            # Reshape back to (B, T, D)
            audio_acoustic_repr = audio_acoustic_repr.permute(0, 2, 1)  # (B, T, D)
            #print(f"Audio acoustic representation shape after TCN: {audio_acoustic_repr.shape}")
            audio_acoustic_attended = self.cross_attention_audio_acoustic(audio_acoustic_repr, context)
            # Reshape to (B, T, D)
            features.append(audio_acoustic_attended)  # Mean pooling over time dimension

        if self.modalities.get('audio_semantic', False):
            # Reshape to (B, D, T)
            audio_semantic_emb = audio_semantic_emb.permute(0, 2, 1)  # (B, D, T)
            # Apply TCN
            audio_semantic_repr = self.audio_semantic_tcn(audio_semantic_emb)
            # Reshape to (B, T, D)
            audio_semantic_repr = audio_semantic_repr.permute(0, 2, 1)  # (B, T, D)
            audio_semantic_attended = self.cross_attention_audio_semantic(audio_semantic_repr, context)
            features.append(audio_semantic_attended)  # Mean pooling over time dimension

        if self.modalities.get('visual', False):
            # Reshape to (B, D, T)
            visual_emb = visual_emb.permute(0, 2, 1)  # (B, D, T)
            visual_repr = self.visual_tcn(visual_emb)
            #print(f"Visual representation shape after TCN: {visual_repr.shape}")
            # Reshape to (B, T, D)
            visual_repr = visual_repr.permute(0, 2, 1)  # (B, T, D)
            visual_attended = self.cross_attention_visual(visual_repr, context)
            features.append(visual_attended)  # Mean pooling over time dimension

        # Concatenate features along last dim
        if not features:
            raise ValueError("No modality is enabled!")
        
        
        # features.append(clip_description_emb)  # Add context features
        concatenated = torch.cat(features, dim=-1)  # (B, D_total)

        

        # Pool over time dimension (e.g., mean pooling)
        # pooled = concatenated.mean(dim=1)  # (B, D_total)

        # MLP prediction
        logits = self.mlp(concatenated)  # (B, num_reaction_classes)

        return logits
#%%
# Example usage
if __name__ == "__main__":
    #%%
    # Define the model
    embedding_dims = {
        'audio_acoustic': 768,
        'audio_semantic': 768,
        'visual': 768,
        'clip_description': 768
    }
    num_genre = 10
    num_reaction_classes = 21
    tcn_channel = {
        'audio_acoustic': [128, 128],
        'audio_semantic': [128, 128],
        'visual': [128, 128]
    }
    tcn_attention = 0
    kernel_size = 3
    modalities = {
        'audio_acoustic': True,
        'audio_semantic': True,
        'visual': True,
        'clip_description': True,
        'genre': True
    }
    context_attention_hidden_dim = 128

    model = MultimodalReactionPredictor(embedding_dims, num_genre, num_reaction_classes, tcn_channel, tcn_attention, kernel_size, modalities, context_attention_hidden_dim=context_attention_hidden_dim)
    
    # Dummy input data
    audio_acoustic_emb = torch.randn(32, 100, embedding_dims['audio_acoustic'])
    audio_semantic_emb = torch.randn(32, 100, embedding_dims['audio_semantic'])
    visual_emb = torch.randn(32, 200, embedding_dims['visual'])
    clip_description_emb = torch.randn(32, embedding_dims['clip_description'])
    genre = torch.randn(32, num_genre)

    # Forward pass
    logits = model(audio_acoustic_emb, audio_semantic_emb, visual_emb, clip_description_emb, genre)
    
    print(logits.shape)  # Expected output: (32, num_reaction_classes)
# %%
