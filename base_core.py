import os
import json
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tensorboard_logger import Logger
from eval_detection import ANETdetection
from PL_generation import train_filter_proposals
from config import *


def check_performance(props_path,gt_path):   
    tIoU_thresh = np.linspace(0.1, 0.7, 7)        
    anet_detection = ANETdetection(gt_path, props_path,
                                subset="Validation", tiou_thresholds=tIoU_thresh,
                                verbose=False, check_status=False)
    mAP, _ = anet_detection.evaluate()
    print("mAP", mAP, "avg", mAP.mean(axis=0))


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale

def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, vid_fps, feat_stride, window_len, _lambda=0.2, gamma=0.0):
    temp = []
    for i in range(len(tList)):
        c_temp = []
        temp_list = np.array(tList[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])
                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])
                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                start_index = grouped_temp_list[j][0]
                end_index = grouped_temp_list[j][-1] + 1
                start_index = (start_index / scale)
                end_index = (end_index / scale)
                start_frame = start_index * feat_stride + (window_len/2)
                end_frame = end_index * feat_stride + (window_len/2)
                t_start = start_frame/vid_fps  
                t_end = end_frame/ vid_fps  
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp

def result2json(result):
    result_file = []    
    for i in range(len(result)):
        line = {'label': class_dict[result[i][0]], 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file

def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]
    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]
    return keep


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def select_seed(cas_sigmoid_fuse, point_anno,bkg_thresh = 0.95):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)    
    bkg_score = cas_sigmoid_fuse[:,:,-1]
    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1)
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b,:act_idx[0]]
            idx_tmp = bkg_seed[b,:act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1
            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index+1] = 1
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b,act_idx[-1]+1:]
            idx_tmp = bkg_seed[b,act_idx[-1]+1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1
            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = 1
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = 1
        for i in range(len(act_idx) - 1):
            if act_idx[i+1] - act_idx[i] <= 1:
                continue
            bkg_score_tmp = bkg_score[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp = bkg_seed[b,act_idx[i]+1:act_idx[i+1]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = 1
            if idx_tmp.sum() >= 2:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index+1:end_index] = 1                                   
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = 1
    return bkg_seed

class ThumosFeature(data.Dataset):
    def __init__(self, data_path, feat_stride, feat_offset, mode, seed=-1):
        if seed >= 0:
            set_seed(seed)
        self.feat_stride= feat_stride
        self.feat_offset= feat_offset
        self.mode = mode        
        self.feature_path = data_path+ "i3d_features"
        with open(data_path+ 'fps_dict.json', 'r') as fid:
            self.fps_dict = json.load(fid)
        self.point_anno = pd.read_csv(os.path.join(data_path, 'point_gaussian', 'point_labels.csv'))
        anno_path = data_path+"annotations/thumos14.json"
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()
        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()
        self.class_name_to_idx = dict((v, k) for k, v in class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())
    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        vid_name = self.vid_list[index]
        feature= np.load(os.path.join(self.feature_path,vid_name + '.npy')).astype(np.float32)
        feature = torch.from_numpy(feature)
        vid_num_seg = feature.shape[0]
        self.video_fps = self.anno['database'][vid_name]["fps"]
        anno_list = self.anno['database'][vid_name]['annotations']
        label = np.zeros([self.num_classes], dtype=np.float32)
        classwise_anno = [[]] * self.num_classes
        for _anno in anno_list:
            label[self.class_name_to_idx[_anno['label']]] = 1
            classwise_anno[self.class_name_to_idx[_anno['label']]].append(_anno)
        temp_anno = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)

        temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'class_index']]
        for key in temp_df['point'].keys():            
            class_idx = temp_df['class_index'][key]
            ##################################################################
            point = temp_df['point'][key]
            frame_2sec= point/float(self.fps_dict[vid_name])
            sec_2frame = frame_2sec*self.video_fps
            point_indx = sec_2frame/ self.feat_stride- self.feat_offset 
            point_indx = int(point_indx) 
            point_indx = min(point_indx, vid_num_seg-1)
            ##################################################################
            temp_anno[point_indx][class_idx] = 1
        temp_anno = torch.from_numpy(temp_anno)

        return index, feature, label, temp_anno, self.vid_list[index], vid_num_seg, self.video_fps


class Cls_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Cls_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,stride=1, padding=1),                      
            nn.ReLU())
        self.classifier = nn.Sequential(nn.Conv1d(in_channels=2048, out_channels=num_classes+1, kernel_size=1,stride=1, padding=0, bias=False))
        self.drop_out = nn.Dropout(p=0.7)
    def forward(self, x):
        out = x.permute(0, 2, 1)
        out = self.conv_1(out)
        out = self.drop_out(out)
        cas = self.classifier(out)
        cas = cas.permute(0, 2, 1)
        return cas
    
class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.r_act = r_act
        self.cls_module = Cls_Module(len_feature, num_classes)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, vid_labels=None):
        num_segments = x.shape[1]
        k_act = num_segments // self.r_act
        cas = self.cls_module(x)
        cas_sigmoid = self.sigmoid(cas)
        bg_score = cas_sigmoid[:,:,-1].unsqueeze(2)
        fg_score = 1 - cas_sigmoid[:,:,-1].unsqueeze(2)
        cas_sigmoid_fg = cas_sigmoid[:,:,:-1] * fg_score
        cas_sigmoid_fg = torch.cat((cas_sigmoid_fg, bg_score), dim=2)
        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        topk_scores = value[:,:k_act,:-1]
        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1)
        else:
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (torch.mean(cas_sigmoid[:,:,:-1], dim=1) * (1 - vid_labels))
        return vid_score, cas_sigmoid_fg


class Total_loss(nn.Module):
    def __init__(self):
        super(Total_loss, self).__init__()
        self.ce_criterion = nn.BCELoss(reduction='none')
    def forward(self, vid_score, cas_sigmoid_fuse, label, point_anno):
        loss = {}
        #################################################################################################################
        loss_vid = self.ce_criterion(vid_score, label)
        loss_vid = loss_vid.mean()        
        #################################################################################################################
        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).cuda()), dim=2)       
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)
        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
        #################################################################################################################
        bkg_seed = select_seed(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())            
        bkg_seed = bkg_seed.unsqueeze(-1).cuda()
        point_anno_bkg = torch.zeros_like(point_anno).cuda()
        point_anno_bkg[:,:,-1] = 1
        indices = torch.nonzero(bkg_seed[0,:,0] == 1).squeeze(-1) 
        point_anno_bkg = point_anno_bkg[:,indices,:] 
        cas_sigmoid_fuse = cas_sigmoid_fuse[:,indices,:]
        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg)).sum(dim=2).mean()
        #################################################################################################################
        loss_total = loss_vid + 0.5 * loss_frame + loss_frame_bkg 
        loss["loss_vid"] = loss_vid
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_total"] = loss_total
        return loss_total, loss


def train(config, net, loader_iter, optimizer, criterion, logger, step):
    net.train()
    total_loss = {}
    total_cost = []
    optimizer.zero_grad()
    for _b in range(config.batch_size):
        _, _data, _label, _point_anno, _, _,_ = next(loader_iter)
        _data = _data.cuda()
        _label = _label.cuda()
        _point_anno = _point_anno.cuda()
        vid_score, cas_sigmoid_fuse = net(_data, _label)            
        cost, loss = criterion(vid_score, cas_sigmoid_fuse, _label, _point_anno)
        total_cost.append(cost)
        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []
            if loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]
    total_cost = sum(total_cost) / config.batch_size
    total_cost.backward()
    optimizer.step()
    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)

def test(config, net, test_loader):
    with torch.no_grad():
        net.eval()
        final_res = {}
        final_res['version'] = 'VERSION 1.3'
        final_res['results'] = {}
        final_res['external_data'] = {'used': True, 'details': 'Features from I3D Network'}        
        num_correct = 0.
        num_total = 0.
        load_iter = iter(test_loader)
        for i in range(len(test_loader.dataset)):
            _, _data, _label, _, vid_name, vid_num_seg, vid_fps = next(load_iter)
            _data = _data.cuda()
            _label = _label.cuda()
            vid_num_seg = vid_num_seg[0].cpu().item()         
            num_segments = _data.shape[1]
            vid_score, cas_sigmoid_fuse = net(_data)
            agnostic_score = 1 - cas_sigmoid_fuse[:,:,-1].unsqueeze(2)
            cas_sigmoid_fuse = cas_sigmoid_fuse[:,:,:-1]    
            label_np = _label.cpu().data.numpy()
            score_np = vid_score[0].cpu().data.numpy()
            pred_np = np.zeros_like(score_np)
            pred_np[np.where(score_np < config.class_thresh)] = 0
            pred_np[np.where(score_np >= config.class_thresh)] = 1
            if pred_np.sum() == 0:
                pred_np[np.argmax(score_np)] = 1
            correct_pred = np.sum(label_np == pred_np, axis=1)
            num_correct += np.sum((correct_pred == config.num_classes).astype(np.float32))
            num_total += correct_pred.shape[0]
            cas = cas_sigmoid_fuse    
            pred = np.where(score_np >= config.class_thresh)[0]
            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])
            cas_pred = cas[0].cpu().numpy()[:, pred]
            cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))
            cas_pred = upgrade_resolution(cas_pred, config.scale)            
            proposal_dict = {}
            agnostic_score = agnostic_score.expand((-1, -1, config.num_classes))
            agnostic_score_np = agnostic_score[0].cpu().data.numpy()[:, pred]
            agnostic_score_np = np.reshape(agnostic_score_np, (num_segments, -1, 1))
            agnostic_score_np = upgrade_resolution(agnostic_score_np, config.scale)
            for i in range(len(config.act_thresh_cas)):
                cas_temp = cas_pred.copy()
                zero_location = np.where(cas_temp[:, :, 0] < config.act_thresh_cas[i])
                cas_temp[zero_location] = 0
                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c, 0] > 0)
                    seg_list.append(pos)
                proposals = get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale,vid_fps, config.feat_stride, config.window_len)     
                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[i]
            for i in range(len(config.act_thresh_agnostic)):
                cas_temp = cas_pred.copy()
                agnostic_score_np_temp = agnostic_score_np.copy()
                zero_location = np.where(agnostic_score_np_temp[:, :, 0] < config.act_thresh_agnostic[i])
                agnostic_score_np_temp[zero_location] = 0
                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_np_temp[:, c, 0] > 0)
                    seg_list.append(pos)
                proposals = get_proposal_oic(seg_list, cas_temp, score_np, pred, config.scale, vid_fps, config.feat_stride, config.window_len)
                for i in range(len(proposals)):
                    class_id = proposals[i][0][0]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[i]
            final_proposals = []
            for class_id in proposal_dict.keys():
                final_proposals.append(nms(proposal_dict[class_id], thresh=config.nms_thresh))
            final_proposals = [final_proposals[i][j] for i in range(len(final_proposals)) for j in range(len(final_proposals[i]))]
            final_res['results'][vid_name[0]] = result2json(final_proposals)

        json_path = os.path.join(config.output_path, 'base_proposals.json')
        with open(json_path, 'w') as f:
            json.dump(final_res, f)
            f.close()        


def base_function(args):
    config = BaseConfig(args)
    worker_init_fn = None
    if config.seed >= 0:
        set_seed(config.seed)
        worker_init_fn = np.random.seed(config.seed)
    #############################################################################################################
    # Prepare the dataset
    train_dataset = ThumosFeature(data_path=config.data_path, feat_stride= config.feat_stride, feat_offset = config.feat_offset, mode='train',seed=config.seed)
    train_loader = data.DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=config.num_workers,worker_init_fn=worker_init_fn)                        
    eval_train_loader = data.DataLoader(train_dataset,batch_size=1, shuffle=False, num_workers=config.num_workers,worker_init_fn=worker_init_fn)
    #############################################################################################################
    # Train the model 
    net = Model(config.len_feature, config.num_classes, config.r_act)
    net = net.cuda()
    criterion = Total_loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    logger = Logger(config.log_path)
    for step in tqdm(range(1, config.num_iters + 1),total = config.num_iters,dynamic_ncols = True):
        if (step - 1) % (len(train_loader) // config.batch_size) == 0:
            loader_iter = iter(train_loader)
        train(config, net, loader_iter, optimizer, criterion, logger, step)
    torch.save(net.state_dict(), os.path.join(config.model_path, "base_model.pkl"))
    #############################################################################################################
    # Test the model on the train set for self-training
    test(config, net, eval_train_loader)
    #############################################################################################################
    # Generate pseudo-labels
    input_path = os.path.join(config.output_path, 'base_proposals.json')
    train_filter_proposals(input_path,config.point_dict_path, config.output_path)
    print("Performance of Pseudo-labels on the train set:---------------------")
    check_performance(config.PL_path, config.gt_path)


