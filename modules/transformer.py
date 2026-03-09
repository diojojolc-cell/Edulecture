import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config.base_config import Config

# class MultiHeadedAttention(nn.Module):
#     def __init__(self, config: Config):
#         super(MultiHeadedAttention, self).__init__()
#         self.embed_dim = config.embed_dim
#         self.num_heads = config.num_mha_heads
#         assert self.embed_dim % self.num_heads == 0
#         self.head_dim = self.embed_dim // self.num_heads
        
#         self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
#         self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
#     def forward(self, text_embeds, video_embeds):
#         """
#         Input
#             text_embeds: num_texts x embed_dim
#             video_embeds: num_vids x num_frames x embed_dim
#         Output
#             o: num_vids x num_texts x embed_dim
#         """
#         num_texts, _ = text_embeds.shape
#         # num_texts x embed_dim
#         q = self.q_proj(text_embeds)
#         q = q.reshape(num_texts, self.num_heads, self.head_dim)
#         # num_heads x head_dim x num_texts
#         q = q.permute(1,2,0)

#         num_vids, num_frames, _ = video_embeds.shape
#         # num_vids x num_frames x embed_dim
#         k = self.k_proj(video_embeds)
#         k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
#         # num_vids x num_heads x num_frames x head_dim
#         k = k.permute(0,2,1,3)

#         # num_vids x num_frames x embed_dim
#         v = self.v_proj(video_embeds)
#         v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
#         # num_vids x num_heads x head_dim x num_frames
#         v = v.permute(0,2,3,1)

#         # num_vids x num_heads x num_frames x num_texts
#         attention_logits = k @ q
#         attention_logits = attention_logits / math.sqrt(self.head_dim)
#         attention_weights = F.softmax(attention_logits, dim=2)

#         # num_vids x num_heads x head_dim x num_texts
#         attention = v @ attention_weights
#         # num_vids x num_texts x num_heads x head_dim
#         attention = attention.permute(0,3,1,2)
#         attention = attention.reshape(num_vids, num_texts, self.embed_dim)

#         # num_vids x num_texts x embed_dim
#         o = self.out_proj(attention)
#         return o


# class Transformer(nn.Module):
#     def __init__(self, config: Config):
#         super(Transformer, self).__init__()
#         self.embed_dim = config.embed_dim
#         dropout = config.transformer_dropout

#         self.cross_attn = MultiHeadedAttention(config)

#         self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
#         self.layer_norm1 = nn.LayerNorm(self.embed_dim)
#         self.layer_norm2 = nn.LayerNorm(self.embed_dim)
#         self.layer_norm3 = nn.LayerNorm(self.embed_dim)
#         self.dropout = nn.Dropout(dropout)

#         self._init_parameters()

    
#     def _init_parameters(self):
#         for name, param in self.named_parameters():
#             if 'linear' in name or 'proj' in name:
#                 if 'weight' in name:
#                     nn.init.eye_(param)
#                 elif 'bias' in name:
#                     param.data.fill_(0.)


#     def forward(self, text_embeds, video_embeds):
#         """
#         Input
#             text_embeds: num_texts x embed_dim
#             video_embeds: num_vids x num_frames x embed_dim
#         Output
#             out: num_vids x num_texts x embed_dim
#         """
#         text_embeds = self.layer_norm1(text_embeds)
#         video_embeds = self.layer_norm1(video_embeds)

#         # num_vids x num_texts x embed_dim
#         attn_out = self.cross_attn(text_embeds, video_embeds)
#         attn_out = self.layer_norm2(attn_out)

#         linear_out = self.linear_proj(attn_out)
#         out = attn_out + self.dropout(linear_out)
#         out = self.layer_norm3(out)

#         return out
class Config:
    def __init__(self, embed_dim=512, num_mha_heads=8, transformer_dropout=0.1):
        self.embed_dim = embed_dim
        self.num_mha_heads = num_mha_heads
        self.transformer_dropout = transformer_dropout

class MultiHeadedAttention(nn.Module):
    def __init__(self, config: Config):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = config.embed_dim
        self.num_heads = config.num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, text_embeds, video_embeds):
        """
        Input:
            text_embeds: (b, 1, embed_dim) - 每个batch一个文本Query
            video_embeds: (b, num_frames, embed_dim) - 对应视频特征
        Output:
            o: (b, 1, embed_dim) - 每个样本独立的注意力输出
        """
        b, _, _ = text_embeds.shape
        _, t, _ = video_embeds.shape

        Q = self.q_proj(text_embeds)    # (b, 1, embed_dim)
        K = self.k_proj(video_embeds)   # (b, t, embed_dim)
        V = self.v_proj(video_embeds)   # (b, t, embed_dim)

        # 重塑为多头形式
        Q = Q.view(b, 1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, heads, 1, d)
        K = K.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, heads, t, d)
        V = V.view(b, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (b, heads, t, d)

        # 计算注意力分数 (b, heads, 1, t)
        attention_logits = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=-1)  # (b, heads, 1, t)

        # 加权求和 (b, heads, 1, d)
        attention = torch.matmul(attention_weights, V)

        # 还原形状 (b, 1, embed_dim)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(b, 1, self.embed_dim)
        o = self.out_proj(attention)
        return o

class Transformer(nn.Module):
    def __init__(self, config: Config):
        super(Transformer, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)
        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        """
        Input:
            text_embeds: (b, embed_dim)
            video_embeds: (b, num_frames, embed_dim)
        Output:
            out: (b, embed_dim) - 对每个样本独立的聚合表示，不会互相影响
        """

        # Layer norm
        text_embeds = self.layer_norm1(text_embeds)      # (b, embed_dim)
        video_embeds = self.layer_norm1(video_embeds)    # (b, num_frames, embed_dim)

        # 将文本增加一个维度 (b, 1, embed_dim)，相当于每个batch的文本变成长度为1的序列
        text_embeds = text_embeds.unsqueeze(1)

        # Cross Attention: 每个batch独立运算
        attn_out = self.cross_attn(text_embeds, video_embeds)  # (b, 1, embed_dim)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)   # (b, 1, embed_dim)

        # 去掉多余的维度 (b, embed_dim)
        out = out.squeeze(1)

        return out
    
class Transformer_shuffle(nn.Module):
    def __init__(self, config: Config):
        super(Transformer_shuffle, self).__init__()
        self.embed_dim = config.embed_dim
        dropout = config.transformer_dropout

        self.cross_attn = MultiHeadedAttention(config)
        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, text_embeds, video_embeds):
        """
        Input:
            text_embeds: (b, embed_dim)
            video_embeds: (b, num_frames, embed_dim)
        Output:
            out: (b, embed_dim) - 对每个样本独立的聚合表示，不会互相影响
        """

        # Layer norm
        text_embeds = self.layer_norm1(text_embeds)      # (b, embed_dim)
        video_embeds = self.layer_norm1(video_embeds)    # (b, num_frames, embed_dim)

        # 将文本增加一个维度 (b, 1, embed_dim)，相当于每个batch的文本变成长度为1的序列
        text_embeds = text_embeds.unsqueeze(1)

        # Cross Attention: 每个batch独立运算
        attn_out = self.cross_attn(text_embeds, video_embeds)  # (b, 1, embed_dim)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)   # (b, 1, embed_dim)

        # 去掉多余的维度 (b, embed_dim)
        out = out.squeeze(1)

        return out
