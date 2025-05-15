#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, activate, d_in, d_hidden, d_out, bias):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias)
        self.activation = nn.ReLU() if activate == 'relu' else nn.ELU() if activate == 'elu' else nn.GELU()
    
    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# d_ins=[l, k, d]'s inputlength same for d_hiddens,dropouts
class MLPsBlock(nn.Module):    
    def __init__(self, activate, d_ins, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=False):
        super(MLPsBlock, self).__init__()
        self.mlp_l = MLP(activate, d_ins[0], d_hiddens[0], d_outs[0], bias)
        self.mlp_k = MLP(activate, d_ins[1], d_hiddens[1], d_outs[1], bias)
        self.mlp_d = MLP(activate, d_ins[2], d_hiddens[2], d_outs[2], bias)
        self.dropout_l = nn.Dropout(p=dropouts[0])
        self.dropout_k = nn.Dropout(p=dropouts[1])
        self.dropout_d = nn.Dropout(p=dropouts[2])
        if ln_first:
            self.ln_l = nn.LayerNorm(d_ins[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_ins[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_ins[2], eps=1e-6)
        else:
            self.ln_l = nn.LayerNorm(d_outs[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_outs[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_outs[2], eps=1e-6)

        self.ln_fist = ln_first
        self.res_project = res_project
        if not res_project:
            print(d_ins, d_outs)
            assert d_ins[0]==d_outs[0], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[1]==d_outs[1], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[2]==d_outs[2], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
        else:
            self.res_projection_l = nn.Linear(d_ins[0], d_outs[0], bias=False)
            self.res_projection_k = nn.Linear(d_ins[1], d_outs[1], bias=False)
            self.res_projection_d = nn.Linear(d_ins[2], d_outs[2], bias=False)
    
    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        if mask is not None:
            print("Warning from MLPsBlock: If using mask, d_in should be equal to d_out.")
        if self.ln_fist:
            x = self.forward_ln_first(x, mask)
        else:
            x = self.forward_ln_last(x, mask)
        return x

    def forward_ln_first(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.ln_l(x.permute(0, 2, 3, 1))
        x = self.mlp_l(x, None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        
        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.ln_k(x.permute(0, 1, 3, 2))
        x = self.dropout_k(self.mlp_k(x, None).permute(0, 1, 3, 2))
        x = x + residual_k
        
        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.ln_d(x)
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d

        return x

    def forward_ln_last(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.mlp_l(x.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0) # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        x = self.ln_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.dropout_k(self.mlp_k(x.permute(0, 1, 3, 2), None).permute(0, 1, 3, 2))
        x = x + residual_k
        x = self.ln_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d
        x = self.ln_d(x)

        return x


# d_in=[l,k,d], hiddens, outs = [[l,k,d], [l,k,d], ..., [l,k,d]] for n layers
class MLPEncoder(nn.Module):
    def __init__(self, activate, d_in, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=[False, False, True]):
        super(MLPEncoder, self).__init__()
        assert len(d_hiddens)==len(d_outs)==len(res_project)
        self.layers_stack = nn.ModuleList([
            MLPsBlock(activate=activate, d_ins=d_in if i==0 else d_outs[i-1], d_hiddens=d_hiddens[i], d_outs=d_outs[i], dropouts=dropouts, bias=bias, ln_first=ln_first, res_project=res_project[i])
        for i in range(len(d_hiddens))])

    def forward(self, x, mask=None):
        for enc_layer in self.layers_stack:
            x = enc_layer(x, mask)
        return x

def get_output_dim(features_compose_t, features_compose_k, d_out, t_out, k_out):
    if features_compose_t in ['mean', 'sum']:
        classify_dim = d_out
    elif features_compose_t == 'cat':
        classify_dim = d_out * t_out
    else:
        raise NotImplementedError

    if features_compose_k in ['mean', 'sum']:
        classify_dim = classify_dim
    elif features_compose_k == 'cat':
        classify_dim = classify_dim * k_out
    else:
        raise NotImplementedError
    return classify_dim

class CubeMLP(nn.Module):
    def __init__(self, opt):
        super(CubeMLP, self).__init__()
        d_a, d_v, d_common = opt.d_a, opt.d_v, opt.d_common
        self.encoders = opt.encoders
        self.features_compose_k = opt.features_compose_k
        self.num_class = opt.num_class
        self.d_common = d_common
        self.time_len = opt.time_len

        if self.encoders == 'conv':
            self.conv_a = nn.Conv1d(d_a, d_common, 3, padding=1)
            self.conv_v = nn.Conv1d(d_v, d_common, 3, padding=1)
        elif self.encoders == 'lstm':
            self.rnn_a = nn.LSTM(d_a, d_common, 1, bidirectional=True, batch_first=True)
            self.rnn_v = nn.LSTM(d_v, d_common, 1, bidirectional=True, batch_first=True)
        elif self.encoders == 'gru':
            self.rnn_a = nn.GRU(d_a, d_common, 2, bidirectional=True, batch_first=True)
            self.rnn_v = nn.GRU(d_v, d_common, 2, bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        self.ln_a = nn.LayerNorm(d_common)
        self.ln_v = nn.LayerNorm(d_common)

        self.dropout_a = nn.Dropout(opt.dropout[0])
        self.dropout_v = nn.Dropout(opt.dropout[1])

        self.mlp_encoder = MLPEncoder(
            activate=opt.activate,
            d_in=[opt.time_len, 2, d_common],
            d_hiddens=opt.d_hiddens,
            d_outs=opt.d_outs,
            dropouts=opt.dropout_mlp,
            bias=True,
            ln_first=False,
            res_project=opt.res_project,
        )
        classify_dim = opt.d_outs[-1][0] * opt.d_outs[-1][1] * opt.d_outs[-1][2]
        self.classifier = nn.Linear(classify_dim, self.num_class)

    def forward(self, audio_emb, visual_emb):
        # encode
        if self.encoders in ['lstm', 'gru']:
            audio_out, _ = self.rnn_a(audio_emb)
            visual_out, _ = self.rnn_v(visual_emb)
        else:
            audio_out = self.conv_a(audio_emb.transpose(1, 2)).transpose(1, 2)
            visual_out = self.conv_v(visual_emb.transpose(1, 2)).transpose(1, 2)

        audio_out = torch.stack(torch.split(audio_out, self.d_common, dim=-1), -1).sum(-1)
        visual_out = torch.stack(torch.split(visual_out, self.d_common, dim=-1), -1).sum(-1)
        
        audio_out = self.dropout_a(self.ln_a(audio_out))
        visual_out = self.dropout_v(self.ln_v(visual_out))

        # # match sequence lengths if needed
        # min_len = min(audio_out.shape[1], visual_out.shape[1])
        # audio_out, visual_out = audio_out[:, :min_len], visual_out[:, :min_len]
        # Pad or truncate to self.time_len
        
        av_num_frames = audio_out.shape[1]
        if av_num_frames > self.time_len:
            # Sample self.time_len frames evenly
            selected_frames = torch.linspace(0, av_num_frames - 1, self.time_len).long()
            audio_out = audio_out[:, selected_frames]
            visual_out = visual_out[:, selected_frames]
        elif av_num_frames < self.time_len:
            # Pad with zeros
            audio_out = F.pad(audio_out, (0, 0, 0, self.time_len-audio_out.shape[1], 0, 0), "constant", 0)
            visual_out = F.pad(visual_out, (0, 0, 0, self.time_len-visual_out.shape[1], 0, 0), "constant", 0)
        # print(f"audio_out shape after padding: {audio_out.shape}")
        # print(f"visual_out shape after padding: {visual_out.shape}")
        x = torch.stack([audio_out, visual_out], dim=2)  # [B, L, 2, D]

        x = self.mlp_encoder(x)
        #print(f"x shape after MLP: {x.shape}")

        # according to the paper, they just flatten it
        x = x.view(x.shape[0], -1)  # [B, L * K * D]

        return self.classifier(x)

#%%
if __name__ == "__main__":
    #%%
    embedding_dims = {
        "encoder": "lstm",
        "d_a": 1024,
        "d_v": 768,
        "d_common": 512,
        # "d_ins": [512, 512, 512],
        "d_outs": [[16, 2, 512], [16, 2, 512]],
        "d_hiddens": [[10, 2, 64], [5, 2, 32]],
        "dropout": [0.1, 0.1, 0.1],
        "dropout_mlp": [0.1, 0.1, 0.1],
        "bias": True,
        "ln_first": False,
        "res_project": [False, False, True],
        "features_compose_t": "cat",
        "features_compose_k": "cat",
        "d_out": 512,
        "task": "classification",
        "num_class": 21,
        "encoders": "lstm",  # or 'conv'
        "features_compose_k": "cat",
        "activate": "relu",
        "time_len": 16,
    }
    from types import SimpleNamespace
    from types import SimpleNamespace    
    opts = SimpleNamespace(d_a=1024, d_v=768, d_common=128, 
                           encoders='gru', features_compose_t='cat', features_compose_k='cat',
                            num_class=7, 
            activate='gelu', time_len=16, 
            d_hiddens=[[16, 2, 128],[8, 2, 64],[4, 1, 32]], 
            d_outs=[[2, 2, 128],[2, 2, 128],[2, 2, 2]],
            dropout_mlp=[0.3,0.4,0.5], dropout=[0.3,0.4,0.5,0.6], bias=False, ln_first=False, res_project=[True, True, True]
            )
    print(opts)


    audio_acoustic_emb = torch.randn(32, 20, embedding_dims['d_a'])
    visual_emb = torch.randn(32, 20, embedding_dims['d_v'])
    model = CubeMLP(opts)
    output = model(audio_acoustic_emb, visual_emb)
    print("Output shape:", output.shape)
    
# %%
