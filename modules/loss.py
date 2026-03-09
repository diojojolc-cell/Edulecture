import torch.nn as nn
import torch
import torch.nn.functional as F
from config.base_config import Config

class CLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims, logit_scale):
        """
        Inputs: cosine similarities
            sims: n x n (text is dim-0)
            logit_scale: 1 x 1
        """
        logit_scale = logit_scale.exp()
        logits = sims * logit_scale
        
        t2v_log_sm = F.log_softmax(logits, dim=1)
        t2v_neg_ce = torch.diag(t2v_log_sm)
        t2v_loss = -t2v_neg_ce.mean()

        v2t_log_sm = F.log_softmax(logits, dim=0)
        v2t_neg_ce = torch.diag(v2t_log_sm)
        v2t_loss = -v2t_neg_ce.mean()

        return (t2v_loss + v2t_loss) / 2.0


def l2norm(X, dim=-1, eps=1e-8):
    norm = torch.norm(X, dim=dim, keepdim=True) + eps
    return X / norm

class Sim_vec_Video(nn.Module):
    def __init__(self, embed_dim, opt=None):
        super(Sim_vec_Video, self).__init__()
        self.opt = opt
        self.sub_dim = embed_dim

        self.weights = nn.Embedding(embed_dim, embed_dim)
        self.sim_map = nn.Linear(embed_dim, 1, bias=False)
        self.temp_learnable = nn.Linear(1, 1, bias=False)

        self.init_weight()

    def init_weight(self):
        self.temp_learnable.weight.data.fill_(4)
        self.sim_map.weight = nn.init.normal_(self.sim_map.weight, mean=2.5, std=0.1)

    def forward(self, txt, vid):
        b, dim = txt.shape
        _, f, _ = vid.shape

        device = txt.device


        sub_dim_index = torch.arange(0, self.sub_dim, device=device).long()  # [dim]
        weights = torch.sigmoid(self.weights(sub_dim_index))  # [dim, dim]

        joint_probability = weights  # [dim, dim]
        mean_probability = torch.mean(joint_probability, dim=1)  # [dim]
        std_probability = torch.std(joint_probability, dim=1)  # [dim]
        thres_probability = mean_probability + self.sim_map.weight.squeeze() * std_probability  # [dim]

        thres_probability = thres_probability.unsqueeze(0).expand(dim, -1)  # [dim, dim]
        values = torch.exp(self.temp_learnable.weight) * (joint_probability - thres_probability)  # [dim, dim]
        mask_probability = torch.tanh(torch.exp(values))  # [dim, dim]
        Dim_learned_weights = mask_probability * weights  # [dim, dim]
        Dim_learned_weights = l2norm(Dim_learned_weights, dim=1)  # [dim, dim]

        Diagonal = Dim_learned_weights.sum(0)  # [dim]
        # Diagonal_Mask = torch.diag_embed(Diagonal)  # [dim, dim]

        txt_expanded = txt.unsqueeze(1).unsqueeze(1)  # [b, 1, 1, dim]
        vid_expanded = vid.unsqueeze(0)  # [1, b, f, dim]

        txt_broadcast = txt_expanded.expand(b, b, f, dim)  # [b, b, f, dim]
        vid_broadcast = vid_expanded.expand(b, b, f, dim)  # [b, b, f, dim]

        txt_vid_product = txt_broadcast * vid_broadcast  # [b, b, f, dim]
        sim_scores_f = (txt_vid_product * Diagonal).sum(dim=-1)  # [b, b, f]

        sim_all_f = sim_scores_f  # [b, b, f]
        sim_all = sim_all_f.mean(dim=-1)  # [b, b]

        sim_all_f = l2norm(sim_all_f, dim=-1)  # [b, b, f]
        sim_all = l2norm(sim_all, dim=-1)  # [b, b]

        return sim_all
    

class LossFactory:
    @staticmethod
    def get_loss(config_loss):
        if config_loss == 'clip':
            return CLIPLoss()
        # elif config_loss == 'softmax_margin_loss':
        #     return  SoftmaxMarginLoss()
        else:
            raise NotImplemented
