import torch
import torch.nn as nn


class process_video_sequence(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768, max_tokens=12):
        super(process_video_sequence, self).__init__()

        self.generator = TemporalTokenGenerator(
            input_dim=input_dim, 
            hidden_dim=hidden_dim, 
            output_dim=output_dim, 
            max_tokens=max_tokens
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_tokens = max_tokens

    def aggregate_tokens(self, all_tokens):
        """
        """
        v_tokens = torch.cat([tokens["v"].unsqueeze(1) for tokens in all_tokens], dim=1)  # [b, num_frames, output_dim]
        c_tokens = torch.cat([tokens["c"].unsqueeze(1) for tokens in all_tokens], dim=1)  # [b, num_frames, output_dim]
        return {"v": v_tokens, "c": c_tokens}

    def forward(self, video_frames, text_features):
        """
        """
        batch_size, num_frames, dim = video_frames.shape
        device = video_frames.device

        # Initialize token and state queues
        token_queue = {
            "v": torch.zeros((batch_size, 0, dim)).to(device),  # [b, t, input_dim]
            "c": torch.zeros((batch_size, 0, dim)).to(device)   # [b, t, input_dim]
        }
        state_queue = {
            "v": torch.zeros((batch_size, self.hidden_dim)).to(device),  # [b, hidden_dim]
            "c": torch.zeros((batch_size, self.hidden_dim)).to(device)  # [b, hidden_dim]
        }

        all_tokens = []

        for frame_idx in range(num_frames):
            frame_features = {
                "v": video_frames[:, frame_idx, :],  # [b, input_dim]
                "c": text_features                   # [b, input_dim]
            }

            # Forward through TemporalTokenGenerator
            current_tokens, token_queue, state_queue = self.generator(
                frame_features,
                token_queue,
                state_queue
            )
            all_tokens.append(current_tokens)

        # Aggregate tokens across all frames
        return self.aggregate_tokens(all_tokens)
class TemporalTokenGenerator(nn.Module):
    def __init__(self, 
                 input_dim=1536,
                 hidden_dim=512,
                 output_dim=512,
                 max_tokens=8
                 ):
        super(TemporalTokenGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_tokens = max_tokens

        self.conv1d = nn.Conv1d(in_channels=input_dim, 
                                out_channels=hidden_dim, 
                                kernel_size=1,
                                groups=1)

        self.A_v = nn.Linear(hidden_dim, hidden_dim)
        self.B_v = nn.Linear(hidden_dim, hidden_dim)
        self.C_v = nn.Linear(hidden_dim, output_dim)
        self.D_v = nn.Linear(input_dim, output_dim)

        self.A_c = nn.Linear(hidden_dim, hidden_dim)
        self.B_c = nn.Linear(hidden_dim, hidden_dim)
        self.C_c = nn.Linear(hidden_dim, output_dim)
        self.D_c = nn.Linear(input_dim, output_dim)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                          num_heads=4, 
                                          batch_first=True)

    def update_token_queue(self, prev_tokens, current_tokens, max_tokens):
        """
        """
        updated_queue = {}
        for modality in prev_tokens:
            # [b, t, dim]
            old_q = prev_tokens[modality]
            # [b, dim] -> [b, 1, dim]
            new_q = current_tokens[modality].unsqueeze(1)
            cat_q = torch.cat([old_q, new_q], dim=1)
            if cat_q.size(1) > max_tokens:
                cat_q = cat_q[:, -max_tokens:, :]
            updated_queue[modality] = cat_q
        return updated_queue

    def forward(self, current_features, prev_tokens, prev_states):
        """
        """

        cur_v = current_features["v"].unsqueeze(1)  # [b, 1, input_dim]
        x_v = torch.cat((prev_tokens["v"], cur_v), dim=1)  # [b, t+1, input_dim]

        cur_c = current_features["c"].unsqueeze(1)  # [b, 1, input_dim]
        x_c = torch.cat((prev_tokens["c"], cur_c), dim=1)  # [b, t+1, input_dim]

        h_v = self.conv1d(x_v.permute(0, 2, 1))  # => [b, hidden_dim, t+1]
        h_c = self.conv1d(x_c.permute(0, 2, 1))  # => [b, hidden_dim, t+1]

        h_v_flat = h_v.permute(0, 2, 1)  # [b, t+1, hidden_dim]
        h_c_flat = h_c.permute(0, 2, 1)  # [b, t+1, hidden_dim]

        # v attends to c
        h_v_attn, _ = self.attn(query=h_v_flat, 
                                key=h_c_flat, 
                                value=h_c_flat)
        # c attends to v
        h_c_attn, _ = self.attn(query=h_c_flat, 
                                key=h_v_flat, 
                                value=h_v_flat)

        # 4) 用 average pooling“当前时刻”的上下文
        h_v_cur = h_v_attn.mean(dim=1)  # [b, hidden_dim]
        h_c_cur = h_c_attn.mean(dim=1)  # [b, hidden_dim]

        h_v_prev = prev_states["v"]  # [b, hidden_dim]
        h_c_prev = prev_states["c"]  # [b, hidden_dim]

        h_v_new = self.A_v(h_v_prev) + self.B_v(h_v_cur)  # [b, hidden_dim]
        h_c_new = self.A_c(h_c_prev) + self.B_c(h_c_cur)  # [b, hidden_dim]

        x_v_cur_squeezed = cur_v.squeeze(1)  # => [b, input_dim]
        x_c_cur_squeezed = cur_c.squeeze(1)  # => [b, input_dim]

        y_v_new = self.C_v(h_v_new) + self.D_v(x_v_cur_squeezed)  # [b, output_dim]
        y_c_new = self.C_c(h_c_new) + self.D_c(x_c_cur_squeezed)  # [b, output_dim]

        current_tokens = {
            "v": y_v_new,  # [b, output_dim]
            "c": y_c_new
        }

        # 7) 更新 token 队列 (存的是原始输入还是网络输出，由你的需求决定，这里存原始输入当作“历史”)
        #    如果你想存输出，可以改成存 y_v_new, y_c_new
        updated_queue = self.update_token_queue(
            prev_tokens,
            {"v": x_v_cur_squeezed, "c": x_c_cur_squeezed},
            self.max_tokens
        )

        updated_states = {
            "v": h_v_new,  # [b, hidden_dim]
            "c": h_c_new
        }
        return current_tokens, updated_queue, updated_states
    
    


