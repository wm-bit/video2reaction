import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math

class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value), attention


class MultiHeadAttentionOp(nn.Module):

    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.
        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttentionOp, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y, attn = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y, attn

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )

class NonLocalBlock(nn.Module):
    def __init__(self, dim_in=2048, dim_out=2048, dim_inner=256):
        super(NonLocalBlock, self).__init__()

        self.dim_in = dim_in
        self.dim_inner = dim_inner
        self.dim_out = dim_out

        self.theta = nn.Linear(dim_in, dim_inner)
        self.phi = nn.Linear(dim_in, dim_inner)
        self.g = nn.Linear(dim_in, dim_inner)

        self.out = nn.Linear(dim_inner, dim_out)
        self.bn = nn.BatchNorm1d(dim_out)
        self.alpha=nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        residual = x

        batch_size,seq = x.shape[:2]
        x=x.view(batch_size*seq,-1)

        theta = self.theta(x)
        phi = self.phi(x)
        g = self.g(x)

        theta,phi,g=theta.view(batch_size,seq,-1).transpose(1,2).contiguous(),phi.view(batch_size,seq,-1).transpose(1,2).contiguous(),g.view(batch_size,seq,-1).transpose(1,2).contiguous()

        theta_phi = torch.bmm(theta.transpose(1, 2), phi)  # (8, 16, 784) * (8, 1024, 784) => (8, 784, 784)

        theta_phi_sc = theta_phi * (self.dim_inner ** -.5)
        p = F.softmax(theta_phi_sc, dim=-1)

        t = torch.bmm(g, p.transpose(1, 2))
        t = t.transpose(1,2).contiguous().view(batch_size*seq,-1)

        out = self.out(t)
        out = self.bn(out)
        out=out.view(batch_size,seq,-1)

        out = out + self.alpha*residual
        return out


class VAANetErase(nn.Module):
    def __init__(self,
                 n_classes,
                 seq_len,
                 audio_embed_size=1024,
                 visual_embed_size=768
                 ):
        super(VAANetErase, self).__init__()
        self.n_classes = n_classes
        self.seq_len = seq_len

        self.visual_embed_size = visual_embed_size
        self.audio_embed_size = audio_embed_size
        
        self.a_fc = nn.Sequential(
            nn.Linear(audio_embed_size, visual_embed_size),
            nn.BatchNorm1d(visual_embed_size),
            nn.Tanh()
        )
        
        self.drop = nn.Dropout(p=.2)
        self._init_norm_val()
        self._init_hyperparameters()
        self._init_nonlocal()
        self._init_attention_subnets()

    def _init_norm_val(self):
        self.NORM_VALUE = 255.0
        self.MEAN = 100.0 / self.NORM_VALUE


    def _init_hyperparameters(self):
        self.hp = {
            'nc': 2048,
            'k': 512,
            'm': 16,
            'hw': 4
        }

    def _init_attention_subnets(self):

        self.ta_net = nn.ModuleDict({
            'conv': nn.Sequential(
                nn.Conv1d(self.visual_embed_size + self.visual_embed_size, 1, 1, bias=False),
                nn.BatchNorm1d(1),
                nn.Tanh(),
            ),
            'fc': nn.Linear(self.seq_len, self.seq_len, bias=True),
            'relu': nn.ReLU()
        })
        self.fc = nn.Linear(self.visual_embed_size + self.visual_embed_size, self.n_classes)

    def _init_module(self, m):
        if isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _init_nonlocal(self):
        self.nl=nn.Sequential(NonLocalBlock(dim_in=self.visual_embed_size, dim_out=self.visual_embed_size))#,NonLocalBlock(),NonLocalBlock())
        self.nl_a=nn.Sequential(NonLocalBlock(dim_in=self.visual_embed_size, dim_out=self.visual_embed_size))#,NonLocalBlock(),NonLocalBlock())
        self.v2a_attn=MultiHeadAttentionOp(in_features=self.visual_embed_size, head_num=8)
        self.a2v_attn=MultiHeadAttentionOp(in_features=self.visual_embed_size, head_num=8)


    def forward(self, input: torch.Tensor):
        v, a = input
        # input.shape=torch.Size([4, 16, 768])
        v=self.nl(v)# B S D
        
        bs, seq_len = a.shape[0], a.shape[1]
        flatten_a = a.reshape(bs * seq_len, -1)
        flatten_a = self.a_fc(flatten_a)  # [4, 16, 1024] -> [4, 16, 768]
        a = flatten_a.reshape(bs, seq_len, -1)
        a = self.nl_a(a)# B S D
        
        v2a, _ = self.v2a_attn(q=a, k=v, v=v)
        a2v, _ = self.a2v_attn(q=v, k=a, v=a)
        v2 = v + v2a
        a2 = a + a2v

        # print(v2.shape, a2.shape)

        output=torch.cat((v2,a2),dim=-1)
        output=output.transpose(1,2).contiguous()
        Ht = self.ta_net['conv'](output)
        Ht = torch.squeeze(Ht, dim=1)
        Ht = self.ta_net['fc'](Ht)
        At = self.ta_net['relu'](Ht)
        gamma = At.view(bs, seq_len)

        output = torch.mul(output, torch.unsqueeze(At, dim=1).repeat(1, self.visual_embed_size + self.visual_embed_size, 1))
        output = torch.mean(output, dim=2)
        output = self.drop(output)
        output = self.fc(output)
        return output, gamma
    

if __name__ == '__main__':
    model=VAANetErase(
        n_classes=21,
        seq_len=16,
    ).cuda()

    # visual=torch.randn(4,16,3,16,112,112).cuda()
    # visual=torch.randn(4,20,3,16,112,112).cuda()
    visual = torch.randn(4, 16, 768).cuda()
    # audio=torch.randn(4,1600,128).cuda()
    # audio=torch.randn(4,2000,128).cuda()  # batch_size, audio_n_segments * 100 (what is this 100)
    audio=torch.randn(4, 16, 1024).cuda()

    output,gamma=model([visual,audio])

    print(output.shape, gamma.shape)