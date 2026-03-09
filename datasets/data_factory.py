import torch
from config.base_config import Config
from datasets.model_transforms import init_transform_dict
from datasets.baseline_dataset import Baselinedataset

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
class DataFactory:

    @staticmethod
    def get_data_loader(config: Config, split_type='train'):
        img_transforms = init_transform_dict(config.input_res)
        train_img_tfms = img_transforms['clip_train']
        test_img_tfms = img_transforms['clip_test']
        
        if config.dataset_name == "baseline":
            if split_type == 'train':
                dataset = Baselinedataset(config, split_type, train_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=config.num_workers)
            else:
                dataset = Baselinedataset(config, split_type, test_img_tfms)
                return DataLoader(dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers)
        elif config.dataset_name == "baseline_distributed":
            if split_type == 'train':
                dataset = Baselinedataset(config, split_type, train_img_tfms)
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=config.world_size,
                    rank=config.rank,
                    shuffle=True,
                    
                )
                return DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    sampler=sampler,
                    num_workers=config.num_workers,
                    drop_last=True # collate_fn=collate_fn
                )
            else:
                dataset = Baselinedataset(config, split_type, test_img_tfms)
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=config.world_size,
                    rank=config.rank,
                    shuffle=False,
                    
                )
                return DataLoader(
                    dataset,
                    batch_size=config.batch_size,
                    sampler=sampler,
                    num_workers=config.num_workers,
                    drop_last=True # collate_fn=collate_fn
                )
        