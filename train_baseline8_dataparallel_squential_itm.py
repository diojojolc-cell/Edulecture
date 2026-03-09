import os
import torch
import random
import numpy as np
from modules.loss import LossFactory
from config.all_config import gen_log
from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from trainer.trainer_baseline_8_squential_itm import Trainer_baseline_8
from modules.metrics import t2v_metrics, v2t_metrics
from modules.optimization import AdamW, get_cosine_schedule_with_warmup
from modules.loss import Sim_vec_Video
from torch.nn.parallel import DataParallel

import warnings
warnings.filterwarnings("ignore")

# @WJM: solve num_workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def main():

    # config
    config = AllConfig()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    writer = None

    # GPU
    if config.gpu is not None and config.gpu != '99':
        print('set GPU')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception('NO GPU!')

    # @WJM: add log
    msg = f'model pth = {config.model_path}'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    msg = f'\nconfig={config.__dict__}\n'
    gen_log(model_path=config.model_path, log_name='log_trntst', msg=msg)
    gen_log(model_path=config.model_path, log_name='log_trntst', msg='record all training and testing results')
    gen_log(model_path=config.model_path, log_name='log_tot_loss', msg='Prepare to record loss values per batch ')
    gen_log(model_path=config.model_path, log_name='log_ori_loss', msg='Prepare to record loss values per batch ')
    gen_log(model_path=config.model_path, log_name='log_sup_loss', msg='Prepare to record loss values per batch ')

    # seed
    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config.arch=="baseline_8_squential_itm":
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained("/home/pc/liuc/clip-vit-base-patch32", TOKENIZERS_PARALLELISM=False)
    elif config.arch=="baseline_8_squential_itm_ch":
    # chinese
        from transformers import ChineseCLIPProcessor
        processor = ChineseCLIPProcessor.from_pretrained("/share/home/xscow_jnugql/liuc/mutil_grade/chinese-clip-vit-base-patch16")
        tokenizer = processor.tokenizer

    # data I/O
    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config)
    model = DataParallel(model).to(device)

    # metric
    # if config.metric == 't2v':
    #     metrics = t2v_metrics
    # elif config.metric == 'v2t':
    #     metrics = v2t_metrics
    # else:
    #     raise NotImplemented


    # for name, param in model.named_parameters():
    #     print(f"Parameter name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    params_optimizer = list(model.named_parameters())
    clip_params = [p for n, p in params_optimizer if "clip." in n]
    noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    
    optimizer_grouped_params = [
        {'params': clip_params, 'lr': config.clip_lr},
        {'params': noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.num_epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    
    loss = LossFactory.get_loss(config.loss)

    trainer = Trainer_baseline_8(model=model,
                      optimizer=optimizer,
                      loss=loss,
                      config=config,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      tokenizer=tokenizer,
                      )


    trainer.train()


if __name__ == '__main__':
    main()
