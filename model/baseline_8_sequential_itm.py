
import torch
import numpy as np
import random
import torch.nn as nn
from torch.nn import functional as F
from config.base_config import Config
from modules.crossattention import ca, ca_audio
from transformers import CLIPModel, CLIPVisionModel
from modules.transformer import Transformer
from modules.squential import process_video_sequence

class Baseline_8_sequential(nn.Module):
    def __init__(self, config: Config):
        super(Baseline_8_sequential, self).__init__()
        self.config = config
        if config.clip_arch == 'ViT-B/32':
            self.clip_1 = CLIPModel.from_pretrained(
                "/home/pc/liuc/clip-vit-base-patch32"
            )
            del self.clip_1.vision_model
            self.clip_2 = CLIPVisionModel.from_pretrained(
                "/home/pc/liuc/clip-vit-base-patch32"
            )
        else:
            raise ValueError("Unsupported CLIP architecture specified in config.")

        # self.pool_frames = Transformer(config)
        # self.cross_attention_vv = ca_audio(config)
        self.process_video_sequence = process_video_sequence()
        self.itm_classifier = ITMClassifier(input_dim=768, hidden_dim=256, dropout_prob=0.1)

        self.linear_proj_1 = nn.Linear(512, 768)

        self.logit_scale_1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.logit_scale_2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, data, is_train=True):

        video_data = data['video']
        text_data = data['text']
        b, f, _, _, _ = video_data.shape

        video_data = video_data.reshape(-1, 3, self.config.input_res, self.config.input_res)
        text_features = self.clip_1.get_text_features(**text_data)
        text_features_expand = self.linear_proj_1(text_features)

        video_features = self.clip_2(video_data, output_hidden_states=True)
        video_last_features = video_features.hidden_states[-1].mean(1)
        original_video_features = video_last_features.reshape(b, f, -1)
        original_video_features_pooled = original_video_features.mean(1)

        interaction_outputs = self.process_video_sequence(
            original_video_features, 
            text_features_expand
        )
        semantic_mask_base = interaction_outputs['c']
        final_semantic_mask = torch.sigmoid(semantic_mask_base)
        masked_video_features = original_video_features * final_semantic_mask

        masked_video_features_pooled = torch.max(masked_video_features, dim=1).values

        if is_train:
            fused_features_pos = interaction_outputs['v']
            fused_features_pos_pooled = fused_features_pos.mean(dim=1)

            with torch.no_grad():
                neg_indices = []
                for i in range(b):
                    possible_indices = [j for j in range(b) if j != i]
                    if possible_indices:
                        neg_indices.append(random.choice(possible_indices))
                    else:
                        neg_indices.append(i)

                neg_text_features = text_features_expand[neg_indices]

            interaction_outputs_neg = self.process_video_sequence(
                original_video_features,
                neg_text_features
            )
            fused_features_neg = interaction_outputs_neg['v']
            fused_features_neg_pooled = fused_features_neg.mean(dim=1)

            fused_features_all = torch.cat([fused_features_pos_pooled, fused_features_neg_pooled], dim=0)

            labels_pos = torch.ones(b, dtype=torch.long)
            labels_neg = torch.zeros(b, dtype=torch.long)
            labels_all = torch.cat([labels_pos, labels_neg], dim=0).to(fused_features_all.device)

            itm_logits = self.itm_classifier(fused_features_all)

            return text_features_expand, masked_video_features_pooled, original_video_features_pooled, self.logit_scale_1, itm_logits, labels_all
        # if is_train:
        #     return text_features_expand, masked_video_features_pooled, original_video_features_pooled, self.logit_scale_1,
        else:
            return text_features_expand, original_video_features_pooled, self.logit_scale_1.exp()

class ITMClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout_prob=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_dim, 2)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)
