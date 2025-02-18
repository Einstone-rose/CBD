class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 9
        self.kernel_size = 8
        self.stride = 1
        self.final_out_channels = 128

        self.num_classes = 6
        self.dropout = 0.35
        self.features_len = 18
        
        self.seq_len = 128
        self.encoder_layer_num = 2  # 8
        self.encode_head_num = 2  # 4
        self.decode_layer_num = 1
        self.decode_head_num = 1
        self.freq_decode_layer = 2
        self.mask_ratio = 0.75
        self.lam = 0.1

        # training configs
        self.num_epoch = 100
        self.finetune_epoch = 200
        self.supervised_epoch = 100

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 1e-4
        self.ft_lr = 1e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 128  # 100
        self.timesteps = 6