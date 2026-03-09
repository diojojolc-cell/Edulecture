import os
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from modules.basic_utils import load_json
from torch.utils.data import Dataset
from config.base_config import Config
from datasets.video_capture import VideoCapture


class Baselinedataset(Dataset):

    def __init__(self, config: Config, split_type = 'train', img_transforms=None):
        self.config = config
        self.img_transforms = img_transforms
        self.split_type = split_type

        if config.train_data_path is not None:
            self.db = load_json(config.train_data_path) 
        else:
            print('no train data path')
        if config.test_data_path is not None:
            self.test_db = load_json(config.test_data_path)  
        else:
            print('no test data path')
    
            
    def __getitem__(self, index):

        if self.split_type == 'train':
            item= self.db[index]
        else: # /share/home/xscow_jnugql/liuc/ours_model/dataset/test_data_add_id.json
            item= self.test_db[index]

        raw_text = item['caption']
        video_path = os.path.join(
                '/share/home/xscow_jnugql/liuc/mutil_grade/datasets/',
                f"{item['video_path']}"
            )
        if self.split_type == 'train':

            video, idxs = VideoCapture.load_frames_from_video(video_path,
                                                            self.config.num_frames, # 12
                                                            self.config.video_sample_type) #[12, 3, 480, 852]) 
        else :
            video, idxs = VideoCapture.load_frames_from_video(video_path,
                                                            self.config.num_frames, # 12
                                                            sample='uniform') #[12, 3, 480, 852])

        if self.img_transforms is not None:
            video = self.img_transforms(video)

        return {
            'video': video,
            'text': raw_text,
        }

    def __len__(self):
        if self.split_type == 'train':
            return len(self.db)

        return len(self.test_db)


    def _get_vidpath_and_caption_by_index(self, index):
        if self.split_type == 'train':
            vid, caption, senid = self.all_train_pairs[index]
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            return video_path, caption, vid, senid
        else:
            vid = self.test_df.iloc[index].video_id
            video_path = os.path.join(self.videos_dir, vid + '.mp4')
            caption = self.test_df.iloc[index].sentence
            return video_path, caption, vid

    
    def _construct_all_train_pairs(self):
        self.all_train_pairs = []
        if self.split_type == 'train':
            for vid in self.train_vids:
                for caption, senid in zip(self.vid2caption[vid], self.vid2senid[vid]):
                    self.all_train_pairs.append([vid, caption, senid])
            
    def _compute_vid2caption(self):
        self.vid2caption = defaultdict(list)
        self.vid2senid   = defaultdict(list)

        for annotation in self.db['sentences']:
            caption = annotation['caption']
            vid = annotation['video_id']
            self.vid2caption[vid].append(caption)
            senid = annotation['sen_id']
            self.vid2senid[vid].append(senid)
