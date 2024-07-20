import os
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from main_model import *
from main_utils import (train_one_epoch, save_checkpoint, make_optimizer, make_scheduler, fix_random_seed, trivial_batch_collator, worker_init_reset_seed, ModelEma, save_best_record_thumos)
from config import *
from eval_detection import ANETdetection


def truncate_feats(feats, video_item , feat_stride, feat_offset, feature_fps, max_seq_len, crop_ratio=None, max_num_trials=500):

    # feats: C x T , 'points': N x 1, 'labels': N          

    feat_len = feats.shape[1]
    num_points = video_item["points"].shape[0]
    if feat_len <= max_seq_len: # max_seq_len : randomly picked from [0.9T, T]        
        max_seq_len = random.randint(max(round(crop_ratio[0] * feat_len), 1), min(round(crop_ratio[1] * feat_len), feat_len),)            
    ######################################################################
    cutoff_thresh = num_points * 0.7 
    for _ in range(max_num_trials):
        st = random.randint(0, feat_len - max_seq_len)
        ed = st + max_seq_len
        final_point_list= []
        final_label_list = []
        final_proposal_list = []
        for instance_idx in range(num_points):
            curr_point = video_item["points"][instance_idx] #in seconds 
            curr_point_indx = (curr_point*feature_fps) /feat_stride- feat_offset  #feature grid 
            #------------------------------------------------------------------------------
            curr_start, curr_end = video_item["proposals"][instance_idx] #in seconds 
            curr_start_indx = (curr_start*feature_fps) /feat_stride- feat_offset  #feature grid 
            curr_end_indx = (curr_end*feature_fps) /feat_stride- feat_offset  #feature grid 
            #------------------------------------------------------------------------------
            if curr_point_indx <= ed-1 and curr_point_indx >= st :        
                shifted_point =  round(curr_point_indx-st) 
                #-----------------------------------------------------
                shifted_start =  round(curr_start_indx-st) 
                shifted_start = max(0,shifted_start)
                shifted_start = min(shifted_start,max_seq_len-1)
                #-----------------------------------------------------
                shifted_end =  round(curr_end_indx-st) 
                shifted_end = max(0,shifted_end)
                shifted_end = min(shifted_end,max_seq_len-1)
                #-----------------------------------------------------
                shifted_proposal = [shifted_start,shifted_end] #feature grid 
                #-----------------------------------------------------
                final_point_list.append(shifted_point) #feature grid 
                final_proposal_list.append(shifted_proposal) #feature grid 
                final_label_list.append(video_item['labels'][instance_idx])
            #------------------------------------------------------------------------------
        if len(final_point_list)> cutoff_thresh:
            break 
    ######################################################################
    feats = feats[:, st:ed].clone()  
    final_points = np.asarray(final_point_list, dtype=np.float32)  # N x 1 
    final_labels = np.asarray(final_label_list)  # N x 1 
    final_proposals = np.asarray(final_proposal_list, dtype=np.float32)  # N x 1 
    return [feats, final_points, final_labels, final_proposals]



class ThumosFeature(data.Dataset):
    def __init__(self, is_training,args):
        config = MainConfig(args)
        self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())        
        self.is_training =is_training # True #False
        self.feature_fps = config.dataset["feature_fps"]
        self.feat_stride = config.dataset["feat_stride"] 
        self.num_frames = config.dataset["num_frames"] 
        self.feat_offset = config.dataset["feat_offset"]
        self.input_dim = config.dataset["input_dim"] 
        self.downsample_rate = config.dataset["downsample_rate"] 
        self.max_seq_len = config.dataset["max_seq_len"] 
        self.trunc_thresh = config.dataset["trunc_thresh"] 
        self.num_classes = config.dataset["num_classes"] 
        self.crop_ratio = config.dataset["crop_ratio"] 
        self.PL_path = config.PL_path
        self.gt_path = config.gt_path
        self.feat_dir = config.feat_folder
        if is_training:
            self.split = ['train'] 
            self.mode = "train"
            dict_db, label_dict = self._load_json_db_train()
        else:
            self.split= ['test']
            self.mode = "test"
            dict_db, label_dict = self._load_json_db_test()
        self.data_list = dict_db
        self.label_dict = label_dict 

    def _load_json_db_train(self):
        with open(self.PL_path, 'r') as fid:
            pseudo_props_dict = json.load(fid)
        pseudo_props_dict = pseudo_props_dict["results"]
        with open(self.gt_path, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']
        ######################################################################
        label_dict = {}  
        for key, value in json_db.items():
            for act in value['annotations']:
                label_dict[act['label']] = self.class_name_to_idx[act['label']]
        ######################################################################
        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            labels, proposals, points = [], [], []
            for dict_item in pseudo_props_dict[key]:
                label_index = label_dict[dict_item['label']]
                labels.append([label_index])
                proposals.append(dict_item['segment'])
                points.append(dict_item['point'])
            points = np.asarray(points, dtype=np.float32)
            proposals = np.asarray(proposals, dtype=np.float32)
            labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            dict_db += ({'id': key,
                         'proposals': proposals, 
                         'points': points, 
                         'labels' : labels}, )
        return dict_db, label_dict

    def _load_json_db_test(self):
        with open(self.gt_path, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']
        ######################################################################
        label_dict = {}  
        for key, value in json_db.items():
            for act in value['annotations']:
                label_dict[act['label']] = self.class_name_to_idx[act['label']]
        ######################################################################
        dict_db = tuple()
        for key, value in json_db.items():
            if value['subset'].lower() not in self.split:
                continue
            labels= []
            for act in value['annotations']:
                label_index = label_dict[act['label']]
                labels.append([label_index])
            points = np.asarray([], dtype=np.float32)
            labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
            dict_db += ({'id': key,
                         'points': points, 
                         'labels' : labels}, )
        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        #########################################################################
        video_item = self.data_list[index]
        vid_name = video_item['id']
        #########################################################################
        rgb_path = os.path.join(self.feat_dir, self.mode, 'rgb', vid_name + '.npy')
        rgb_feature = np.load(rgb_path).astype(np.float32)
        flow_path = os.path.join(self.feat_dir, self.mode, 'flow', vid_name + '.npy')
        flow_feature = np.load(flow_path).astype(np.float32)
        feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        feats = torch.from_numpy(np.ascontiguousarray(feature.transpose())) # C x T
        #########################################################################
        vid_num_seg = feats.shape[1] 
        #########################################################################
        if self.is_training:
            feats, final_points, final_labels, final_proposals = truncate_feats(feats, video_item , self.feat_stride , self.feat_offset, self.feature_fps, self.max_seq_len, self.crop_ratio)
            #final_points : N x 1 , #final_labels : N x 1             
            vid_num_seg = feats.shape[1]  #updated            
        else:
            final_labels = video_item['labels']
            final_points =  np.asarray([-1])
            final_proposals =  np.asarray([-1])
        #########################################################################
        final_labels = torch.from_numpy(final_labels)
        final_proposals = torch.from_numpy(final_proposals)
        final_points = torch.from_numpy(final_points)
        label_np_arr = np.zeros([self.num_classes], dtype=np.float32)
        for label_index in final_labels:
            label_np_arr[label_index] = 1
        label_np_arr = torch.from_numpy(label_np_arr)
        #########################################################################
        final_dict = {'video_id'        : vid_name,
                     'feats'            : feats,           # C x vid_num_seg
                     'label_np_arr'     : label_np_arr,    # C
                     'final_proposals'     : final_proposals,  # N x 2
                     'final_points'     : final_points,
                     'final_labels'     : final_labels,    # N x 1
                     'vid_num_seg'      : vid_num_seg,
                     'feat_stride'     : self.feat_stride,
                     'feat_num_frames' : self.num_frames}
        return final_dict
    


def train(args):
    config = MainConfig(args)
    ################################################################################
    tb_writer = SummaryWriter(os.path.join(config.model_path, 'logs'))
    rng_generator = fix_random_seed(config.init_rand_seed, include_cuda=True)
    train_dataset = ThumosFeature(is_training = True, args=args)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed),shuffle=True,drop_last=True,generator=rng_generator,persistent_workers=True)
    ################################################################################
    model = Model(args)
    model = nn.DataParallel(model, device_ids=config.devices)
    num_iters_per_epoch = len(train_loader)
    optimizer = make_optimizer(model, config.opt)
    scheduler = make_scheduler(optimizer, config.opt, num_iters_per_epoch)
    model_ema = ModelEma(model)
    ################################################################################
    print("\nStart training the model ...")
    max_epochs = config.opt['epochs'] + config.opt['warmup_epochs']
    for epoch in range(0, max_epochs):
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = config.train_cfg['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=config.print_freq)
        if (((epoch + 1) == max_epochs) or ((config.ckpt_freq > 0) and ((epoch + 1) % config.ckpt_freq == 0))):            
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=config.model_path,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1))
    tb_writer.close()
    print("All done!")
                

def test(args):
    _ = fix_random_seed(0, include_cuda=True)
    config = MainConfig(args)    
    model = Model(args)
    model = nn.DataParallel(model, device_ids=['cuda:0'])
    ckpt_file = config.model_path+"/epoch_0{}.pth.tar".format(config.opt['eval_epochs'])
    checkpoint = torch.load(ckpt_file,map_location = lambda storage, loc: storage.cuda(config.devices[0]))
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint
    #######################################################################
    val_dataset = ThumosFeature(is_training = False,args=args)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1, num_workers=config.num_workers, collate_fn=trivial_batch_collator,
        worker_init_fn=None ,shuffle=False, drop_last=False,generator=None, persistent_workers=True)
    #######################################################################
    num_correct = 0.
    num_total = 0.
    final_res = {}
    final_res['version'] = 'VERSION 1.3'
    final_res['results'] = {}
    final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}
    model.eval()
    for _, video_list in enumerate(val_loader, 0):
        with torch.no_grad():
            is_correct, vid_res, vid_idxs = model(video_list,  is_training=False)
            num_correct += is_correct
            num_total += 1
            final_res['results'][vid_idxs] =vid_res
    test_acc = num_correct / num_total
    #######################################################################    
    proposals_path = os.path.join(config.output_path, 'test_proposals.json')
    with open(proposals_path, 'w') as f:
        json.dump(final_res, f)
        f.close()
    tIoU_thresh = np.linspace(0.1, 0.7, 7)
    anet_detection = ANETdetection(config.gt_path, proposals_path,subset="test", tiou_thresholds=tIoU_thresh,verbose=False, check_status=False)                                                                
    mAP, _ = anet_detection.evaluate()
    #######################################################################
    test_info = {"step": [], "test_acc": [],
                "average_mAP[0.1:0.7]": [], "average_mAP[0.1:0.5]": [], "average_mAP[0.3:0.7]": [],
                "mAP@0.1": [], "mAP@0.2": [], "mAP@0.3": [], "mAP@0.4": [],
                "mAP@0.5": [], "mAP@0.6": [], "mAP@0.7": []}    
    test_info["test_acc"].append(test_acc)
    test_info["average_mAP[0.1:0.7]"].append(mAP[:7].mean())
    test_info["average_mAP[0.1:0.5]"].append(mAP[:5].mean())
    test_info["average_mAP[0.3:0.7]"].append(mAP[2:7].mean())
    for i in range(tIoU_thresh.shape[0]):
        test_info["mAP@{:.1f}".format(tIoU_thresh[i])].append(mAP[i])
    save_best_record_thumos(test_info, os.path.join(config.output_path, "test_results.txt"))
    #######################################################################
    print("average_mAP[0.1:0.7]: ", mAP[:7].mean())
    print("average_mAP[0.1:0.5]: ", mAP[:5].mean())
    print("average_mAP[0.3:0.7]: ", mAP[2:7].mean())
    #######################################################################
    return

