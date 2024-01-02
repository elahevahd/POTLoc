import numpy as np

class BaseConfig(object):
    def __init__(self, args):
        self.num_classes = 20
        self.len_feature = 2048
        self.feature_fps = 25
        self.feat_stride= 4
        self.feat_offset= 2 
        self.window_len = 16
        #########################################
        self.seed = 0
        self.num_workers = 12
        self.num_iters = 4000
        self.lr = 0.0001
        self.batch_size = 16
        #########################################
        self.r_act = 8
        self.scale = 16
        self.class_thresh = 0.7
        self.act_thresh_cas = np.arange(0.15, 0.25, 0.025)
        self.act_thresh_agnostic = np.arange(0.1, 1.0, 0.01)
        self.nms_thresh = 0.5
        #########################################
        self.gt_path = args.gt_path
        self.point_dict_path = args.point_dict_path
        self.data_path = args.data_path
        self.model_path = args.base_model_path
        self.output_path = args.base_output_path
        self.log_path = args.log_path
        self.PL_path = args.PL_path
        #########################################        
    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')


class MainConfig(object):
    def __init__(self, args):
        #########################################
        self.init_rand_seed = 1234567891
        self.devices = ['cuda:0']
        self.batch_size = 3 
        self.num_workers = 6 
        #########################################
        self.print_freq = args.print_freq
        self.ckpt_freq = args.ckpt_freq
        self.gt_path = args.gt_path
        self.PL_path = args.PL_path
        self.feat_folder = args.feat_folder
        self.model_path = args.main_model_path
        self.output_path = args.main_output_path
        #########################################
        self.dataset_name= "thumos"
        self.train_split= ['validation']
        self.val_split= ['test']
        self.dataset = {
            "num_classes": 20, 
            "feat_stride": 4, # temporal stride of the feats            
            "num_frames": 16, # number of frames for each feat
            "feat_offset":2,  #0.5*(num_frames /feat_stride)
            "default_fps": None, # default fps,            
            "input_dim": 2048,
            "len_feature":2048,        
            "r_act": 8,
            "downsample_rate": 1,        
            "max_seq_len": 2304, # max sequence length during training            
            "trunc_thresh": 0.5,   # serve as data augmentation
            "crop_ratio": [0.9, 1.0], # serve as data augmentation       
            "force_upsampling": False, # if true, force upsampling of the input features into a fixed size
            }
        #########################################
        self.opt=  {
            "type": "AdamW",
            "momentum": 0.9,
            "weight_decay": 0.05,
            "learning_rate":  0.0001,           
            "warmup": True,
            "warmup_epochs": 10,
            "epochs": 50,  # excluding the warmup epochs
            "eval_epochs":50,
            "schedule_type": "cosine",
            "schedule_steps": [],
            "schedule_gamma": 0.1,
            }
        #########################################
        self.train_cfg= {
            "center_sample": "radius",
            "sample_radius": 2,
            "train_cls_prior_prob": 0.01,
            "head_empty_cls": [],            
            "init_loss_norm": 100,
            "clip_grad_l2norm": 1.0,
            "label_smoothing": 0.0,
            "loss_weight": 1.0, # on reg_loss
            }
        #########################################
        self.model= {
            "backbone_type": 'convTransformer',
            "fpn_type": "identity",
            "scale_factor": 2, #scale factor between pyramid levels
            "n_head": 4, # number of heads in self-attention        
            "n_mha_win_size": 19, # window size for self attention        
            "embd_kernel_size": 3, # kernel size for embedding network   
            "embd_dim": 512, #feature dim for embedding network     
            "embd_with_ln": True, # if attach group norm to embedding network  
            "fpn_start_level": 0, # starting level for fpn  
            "fpn_dim": 512, # feat dim for FPN
            "fpn_levels":4, #number of fpn levels
            "fpn_with_ln": True, # if add ln at the end of fpn outputs       
            "head_dim": 512,
            "head_kernel_size": 3,
            "head_num_layers": 3,
            "head_with_ln": True,
            }
        #########################################
        self.inference = {
            "act_thresh_agnostic" : np.arange(0.1, 0.9, 0.01),
            "act_thresh_cas": np.arange(0.15, 0.35, 0.025),
            "class_thresh": 0.7,
            "temp_scale": 4,
            "nms_thresh":  0.45
            }
        #########################################
    def __str__(self):
        attrs = vars(self)
        attr_lst = sorted(attrs.keys())
        return '\n'.join("- %s: %s" % (item, attrs[item]) for item in attr_lst if item != 'lr')


class_dict = {0: 'BaseballPitch',
                1: 'BasketballDunk',
                2: 'Billiards',
                3: 'CleanAndJerk',
                4: 'CliffDiving',
                5: 'CricketBowling',
                6: 'CricketShot',
                7: 'Diving',
                8: 'FrisbeeCatch',
                9: 'GolfSwing',
                10: 'HammerThrow',
                11: 'HighJump',
                12: 'JavelinThrow',
                13: 'LongJump',
                14: 'PoleVault',
                15: 'Shotput',
                16: 'SoccerPenalty',
                17: 'TennisSwing',
                18: 'ThrowDiscus',
                19: 'VolleyballSpiking'}