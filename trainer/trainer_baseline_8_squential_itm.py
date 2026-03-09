import torch
import os
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from config.all_config import gen_log
from config.base_config import Config
from collections import defaultdict, deque
from modules.metrics import sim_matrix_training, sim_matrix_it, calculate_recall, calculate_ranks


class Trainer_baseline_8():

    def __init__(self, model, loss, optimizer, config: Config, train_data_loader,
                 valid_data_loader, tokenizer, lr_scheduler=None, writer=None):

        super().__init__()
        self.config = config
        self.device = self._prepare_device()
        self.model = model.to(self.device)
        self.loss = loss.to(self.device)
        self.optimizer = optimizer

        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer 
        # self.Sim_vec_Video = Sim_vec_Video
        # self.audio_encode = audio_encode

        self.start_epoch = 1
        self.global_step = 0

        self.num_epochs = config.num_epochs
        self.writer = writer
        self.checkpoint_dir = config.model_path

        self.log_step = config.log_step
        self.evals_per_epoch = config.evals_per_epoch

        self.pooling_type = config.pooling_type
        self.window_metric = defaultdict(lambda: deque(maxlen=config.eval_window_size))
        self.best_window = -1.0
        self.best = -1.0

    def train(self):
        # for epoch in range(self.start_epoch, self.num_epochs + 1):
        #     result = self._train_epoch(epoch)
        #     res = self._valid_epoch_step(epoch, 0, 0)
        #     if epoch % self.config.save_every == 0:
        #             self._save_checkpoint(epoch, save_best=False)

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            result = self._train_epoch(epoch)

            res = self._valid_epoch_step(epoch, 0, 0)

            if res > self.best:
                print(f"Epoch {epoch}: Validation improved from {self.best:.4f} to {res:.4f}. Saving model...")
                self.best = res
                self._save_checkpoint(epoch, save_best=True)
            else:
                print(f"Epoch {epoch}: Validation did not improve from {self.best:.4f}.")

    def _train_epoch(self, epoch):

        self.model.train()
        total_loss = 0.0
        num_steps = len(self.train_data_loader)
        
        for batch_idx, data in enumerate(self.train_data_loader):
            if self.tokenizer is not None:
                data['text'] = self.tokenizer(data['text'], return_tensors='pt', padding=True, truncation=True, max_length=77)
            if isinstance(data['text'], torch.Tensor):
                data['text'] = data['text'].to(self.device)
            else:
                data['text'] = {key: val.to(self.device) for key, val in data['text'].items()}

            text_feat, masked_video_feat, orig_video_feat, logit_scale, itm_logits, itm_labels = self.model(data, is_train=True)

            logit_scale = logit_scale.mean()
            sim_masked = sim_matrix_training(text_feat, masked_video_feat)
            loss_masked = self.loss(sim_masked, logit_scale)

            sim_orig = sim_matrix_training(text_feat, orig_video_feat)
            loss_orig = self.loss(sim_orig, logit_scale)

            itm_loss = F.cross_entropy(itm_logits, itm_labels)

            loss_all = 0.4 * loss_masked + 0.4 * loss_orig + 0.2 * itm_loss 
            # loss_all = 0.5 * loss_masked + 0.5 * loss_orig 

            self.optimizer.zero_grad()
            loss_all.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            torch.clamp_(logit_scale.data, max=np.log(100))

            self.global_step += 1

            total_loss += loss_all.detach().item()

            if self.config.noloss_record:
                pass
            else:
                gen_log(model_path=self.config.model_path, log_name='log_total_loss',
                        msg=loss_all.item())

            if batch_idx % self.log_step == 0: #| itm_loss:{:.6f}
                msg = ('Train Epoch: {} dl: {}/{} | Total Loss: {:.6f} | loss_masked:{:.6f} | loss_orig :{:.6f} ' .format(
                    epoch,
                    batch_idx,
                    num_steps-1,
                    loss_all.detach().item(),
                    loss_masked.detach().item(),
                    loss_orig.detach().item(),
                    # itm_loss.detach().item(),
                    ))
                gen_log(model_path=self.config.model_path, log_name='log_trntst', msg=msg)

        res = {
            'loss_train':  total_loss / num_steps
        }
        # torch.cuda.empty_cache()
        # print(f"Epoch {epoch}, Batch {batch_idx}, GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        return res

    
    def _valid_epoch_step(self, epoch, step, num_steps):

        self.model.eval()

        all_text_features = []
        all_final_features = []
        all_last_video = []

        with torch.no_grad():
            for batch in tqdm(self.valid_data_loader, desc="Validation"):
                if self.tokenizer is not None:
                    batch['text'] = self.tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=77)
                if isinstance(batch['text'], torch.Tensor):
                    batch['text'] = batch['text'].to(self.device)
                else:
                    batch['text'] = {key: val.to(self.device) for key, val in batch['text'].items()}
                
                # batch['video'] = batch['video'].to(self.device) # Assuming video is also processed and moved to device

                text_features_expand, video_last_features, logit_scale_1, = self.model(batch, is_train=False)

                all_text_features.append(text_features_expand)
                all_last_video.append(video_last_features)

        all_text_features = torch.cat(all_text_features, dim=0)
        all_last_video = torch.cat(all_last_video, dim=0)

        # Calculate similarity matrix
        output_tv = sim_matrix_it(all_text_features, all_last_video)
        sims_tv = output_tv * logit_scale_1.mean()

        # --- Start of Changes ---

        # --- Text-to-Video Evaluation ---
        recalls_tv = calculate_recall(sims_tv)
        ranks_tv = calculate_ranks(sims_tv) # NEW: Calculate ranks
        print("\n--- Text-to-Video Metrics ---")
        print(f"Recall: R@1: {recalls_tv['R1']:.4f}, R@5: {recalls_tv['R5']:.4f}, R@10: {recalls_tv['R10']:.4f}")
        print(f"Ranks:  MeanR: {ranks_tv['MeanR']:.2f}, MedR: {ranks_tv['MedR']:.2f}")

        # --- Video-to-Text Evaluation ---
        sims_vt = sims_tv.t() # Use the transposed matrix for V->T
        recalls_vt = calculate_recall(sims_vt)
        ranks_vt = calculate_ranks(sims_vt) # NEW: Calculate ranks
        print("\n--- Video-to-Text Metrics ---")
        print(f"Recall: R@1: {recalls_vt['R1']:.4f}, R@5: {recalls_vt['R5']:.4f}, R@10: {recalls_vt['R10']:.4f}")
        print(f"Ranks:  MeanR: {ranks_vt['MeanR']:.2f}, MedR: {ranks_vt['MedR']:.2f}\n")

        # Extract all metrics for logging
        r1_tv, r5_tv, r10_tv = recalls_tv['R1'], recalls_tv['R5'], recalls_tv['R10']
        meanr_tv, medr_tv = ranks_tv['MeanR'], ranks_tv['MedR']

        r1_vt, r5_vt, r10_vt = recalls_vt['R1'], recalls_vt['R5'], recalls_vt['R10']
        meanr_vt, medr_vt = ranks_vt['MeanR'], ranks_vt['MedR']

        # Build the updated msg string
        msg = (
            f"validation:  | epoch: {epoch}\n"
            f"Text-Video Recall | R@1: {r1_tv:.4f}, R@5: {r5_tv:.4f}, R@10: {r10_tv:.4f}\n"
            f"Text-Video Ranks  | MeanR: {meanr_tv:.2f}, MedR: {medr_tv:.2f}\n" # NEW: Added ranks log line
            f"Video-Text Recall | R@1: {r1_vt:.4f}, R@5: {r5_vt:.4f}, R@10: {r10_vt:.4f}\n"
            f"Video-Text Ranks  | MeanR: {meanr_vt:.2f}, MedR: {medr_vt:.2f}\n" # NEW: Added ranks log line
        )

        # Store to logs
        gen_log(model_path=self.config.model_path, log_name='validation', msg=msg)
        return r1_tv

    
    # --- End of Changes ---
    def _prepare_device(self):
        """
        setup GPU device if available, move model into configured device
        """
        use_gpu = torch.cuda.is_available()
        device = torch.device('cuda:0' if use_gpu else 'cpu')
        return device
    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param save_best: if True, save checkpoint to 'model_best.pth'
        """

        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")
        else:
            filename = os.path.join(self.checkpoint_dir, 'checkpoint-epoch{}.pth'.format(epoch))
            torch.save(state, filename)
            print("Saving checkpoint: {} ...".format(filename))


    def load_checkpoint(self, model_name):
        """
        Load from saved checkpoints
        :param model_name: Model name experiment to be loaded
        """
        # checkpoint_path = os.path.join(self.checkpoint_dir, model_name)
        # print("Loading checkpoint: {} ...".format(checkpoint_path))

        checkpoint_path = model_name

        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint['epoch'] + 1 if 'epoch' in checkpoint else 1
        state_dict = checkpoint['state_dict']
        
        missing_key, unexpected_key = self.model.load_state_dict(state_dict, strict=False)
        print(f'missing_key={missing_key}')
        print(f'unexpected key={unexpected_key}')

        # if self.optimizer is not None:
        #     self.optimizer.load_state_dict(checkpoint['optimizer'])

        print("Checkpoint loaded")

        