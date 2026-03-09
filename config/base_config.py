from abc import abstractmethod, ABC


class Config(ABC):
    def __init__(self):
        args = self.parse_args()

        # self.it = args.it
        
        self.dataset_name = args.dataset_name
        self.videos_dir = args.videos_dir
        self.msrvtt_train_file = args.msrvtt_train_file
        self.num_frames = args.num_frames
        self.video_sample_type = args.video_sample_type
        self.input_res = args.input_res

        self.exp_name = args.exp_name
        self.model_path = args.model_path 
        self.output_dir = args.output_dir
        self.save_every = args.save_every
        self.log_step = args.log_step
        self.evals_per_epoch = args.evals_per_epoch
        self.load_epoch = args.load_epoch
        self.eval_window_size = args.eval_window_size
        self.metric = args.metric

        self.arch = args.arch
        self.clip_arch = args.clip_arch
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads_iv

        self.loss = args.loss
        self.clip_lr = args.clip_lr
        self.noclip_lr = args.noclip_lr
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.weight_decay = args.weight_decay
        self.warmup_proportion = args.warmup_proportion
    
        self.pooling_type = args.pooling_type
        self.k = args.k
        self.attention_temperature = args.attention_temperature
        self.num_mha_heads = args.num_mha_heads
        self.transformer_dropout = args.transformer_dropout

        self.num_workers = args.num_workers
        self.seed = args.seed
        self.no_tensorboard = args.no_tensorboard
        self.tb_log_dir = args.tb_log_dir

        # @WJM:
        self.stochasic_trials = args.stochasic_trials
        self.datetime = args.datetime
        self.gpu = args.gpu
        self.support_loss_weight = args.support_loss_weight
        self.batch_size_split = args.batch_size_split
        self.chunk_size = args.chunk_size
        self.noloss_record = args.noloss_record
        self.save_memory_mode = args.save_memory_mode
        self.raw_video = args.raw_video
        self.skip_eval = args.skip_eval
        self.DSL = args.DSL
        self.stochastic_prior = args.stochastic_prior
        self.stochastic_prior_std = args.stochastic_prior_std

        self.train_data_path = args.train_data_path
        self.test_data_path = args.test_data_path
        self.select_layer = args.select_layer

        self.text_embedding_path = args.text_embedding_path
        self.vision_embedding_path = args.vision_embedding_path

        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.rank = args.rank

        #layout
        self.vocab_size = 50265
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.type_vocab_size = 2
        self.layer_norm_eps = 1e-5
        self.pad_token_id = 1
        self.max_2d_position_embeddings = 1024
        self.coordinate_size = 128
        self.shape_size = 128

    @abstractmethod
    def parse_args(self):
        raise NotImplementedError

