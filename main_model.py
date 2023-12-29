import math
import numpy as np 
import torch
import torch.nn as nn
from torch.nn import functional as F
from main_backbones import ConvTransformerBackbone, MaskedConv1D, LayerNorm, FPNIdentity, PointGenerator
import main_utils
from config import *


class Total_loss(nn.Module):
    def __init__(self,r_act):
        super(Total_loss, self).__init__()
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.r_act = r_act
    def forward(self, cas_sigmoid_fuse, cas_sigmoid, valid_mask, gt_cls, vid_labels, fpn_intervals):
        cas_sigmoid = cas_sigmoid[:, : , :-1] 
        cls_scores = (cas_sigmoid * valid_mask.unsqueeze(-1)) 
        vid_scores= []
        for interval in fpn_intervals:
            start = interval[0]
            end = interval[1]
            cls_scores_level = cls_scores[:, start:end, :]
            k_act = cls_scores_level.shape[1]//self.r_act 
            value, _ = cls_scores_level.sort(descending=True, dim=1)
            topk_scores = value[:, :k_act,:]  
            mean_topk_score = torch.mean(topk_scores, dim=1) 
            mean_all_scores = torch.mean(cls_scores_level, dim=1) 
            part_1 = mean_topk_score * vid_labels #[B, C] 
            part_2 = mean_all_scores * (1 - vid_labels) #[B, C] 
            vid_score_level = part_1 + part_2
            vid_scores.append(vid_score_level)
        vid_scores = torch.stack(vid_scores, dim=-1) 
        vid_scores = torch.mean(vid_scores, dim=-1)
        loss_vid = self.ce_criterion(vid_scores, vid_labels)
        loss_vid = loss_vid.mean()
        ##########################################################################        
        point_anno = torch.cat((gt_cls, torch.zeros((gt_cls.shape[0], gt_cls.shape[1], 1)).cuda()), dim=2) # point_anno:[B, sum{T_i}, C+1]
        point_anno_masked = (point_anno * valid_mask.unsqueeze(-1)) #[B,T,C+1]
        cas_sigmoid_fuse_masked = (cas_sigmoid_fuse * valid_mask.unsqueeze(-1)) #[B,T,C+1]
        weighting_seq_act = point_anno_masked.max(dim=2, keepdim=True)[0] #[B,T]
        num_actions = point_anno_masked.max(dim=2)[0].sum(dim=1) #[B]
        for b in range(num_actions.shape[0]):
            if num_actions[b].item()==0:
                num_actions[b]=1
        focal_weight_act = (1 - cas_sigmoid_fuse_masked) * point_anno_masked + cas_sigmoid_fuse_masked * (1 - point_anno_masked)
        focal_weight_act = focal_weight_act ** 2
        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse_masked.float(), point_anno_masked.float()) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
        ##########################################################################        
        bkg_seed_list = []
        for interval in fpn_intervals:
            start = interval[0]
            end = interval[1]
            cls_scores_level = cas_sigmoid_fuse_masked[:, start:end, :].detach().cpu() #[B , T_i, C+1]
            gt_target_level = point_anno_masked[:, start:end, :].detach().cpu()   #[B , T_i, C+1]
            bkg_seed = main_utils.select_seed(cls_scores_level, gt_target_level) #[B , T_i]            
            bkg_seed = bkg_seed.unsqueeze(-1).cuda()  #[B , T_i, 1]
            bkg_seed_list.append(bkg_seed)
        bkg_seed_cat = torch.cat(bkg_seed_list, dim=1) #[B , sum{T_i}, 1]
        weighting_seq_bkg = bkg_seed_cat   #[B , sum{T_i}, 1]
        num_bkg = bkg_seed_cat.sum(dim=1)  #[B , 1] or [B]
        for b in range(num_bkg.shape[0]):
            if num_bkg[b].item()==0:
                num_bkg[b]=1
        point_anno_bkg = torch.zeros_like(point_anno).cuda()  #[B , sum{T_i}, C+1]
        point_anno_bkg[:,:,-1] = 1
        focal_weight_bkg = (1 - cas_sigmoid_fuse_masked) * point_anno_bkg + cas_sigmoid_fuse_masked * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse_masked.float(), point_anno_bkg.float()) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
        return loss_vid, loss_frame, loss_frame_bkg


class PtTransformerClsHead(nn.Module):
    """
    1D Conv heads for classification
    """
    def __init__(
        self,
        input_dim,
        feat_dim,
        num_classes,
        prior_prob=0.01,
        num_layers=3,
        kernel_size=3,
        act_layer=nn.ReLU,
        with_ln=False,
        empty_cls = []
    ):
        super().__init__()
        self.act = act_layer()
        self.sigmoid = nn.Sigmoid()
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers-1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(MaskedConv1D(in_dim, out_dim, kernel_size,stride=1,padding=kernel_size//2,bias=(not with_ln)))
            if with_ln:
                self.norm.append(LayerNorm(out_dim))
            else:
                self.norm.append(nn.Identity())

        self.cls_head = MaskedConv1D(feat_dim, num_classes, kernel_size,stride=1, padding=kernel_size//2)                

        if prior_prob > 0: # prior in model initialization to improve stability
            bias_value = -(math.log((1 - prior_prob) / prior_prob))
            torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories. The weights assocaited with these categories will remain unchanged. We set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        assert len(fpn_feats) == len(fpn_masks)
        # apply the classifier for each pyramid level
        out_logits = tuple()
        out_logits_fuse = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            ############################################
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            ############################################
            cur_logits, _ = self.cls_head(cur_out, cur_mask) #[B, C+1, T_i]
            cur_logits = cur_logits.permute(0, 2, 1) #[B, T_i, C+1]
            cas_sigmoid = self.sigmoid(cur_logits)
            cas_sigmoid_fuse = cas_sigmoid[:,:,:-1] * (1 - cas_sigmoid[:,:,-1].unsqueeze(2))
            cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:,:,-1].unsqueeze(2)), dim=2)
            #cas_sigmoid_fuse: [B, T_i, C+1]
            out_logits += (cas_sigmoid, )
            out_logits_fuse += (cas_sigmoid_fuse, )
        return out_logits, out_logits_fuse



class Cls_Module(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(Cls_Module, self).__init__()
        self.len_feature = len_feature
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=self.len_feature, kernel_size=3,stride=1, padding=1),                      
            nn.ReLU()
        )
        self.classifier = nn.Sequential(nn.Conv1d(in_channels=self.len_feature, out_channels=num_classes+1, kernel_size=1,stride=1, padding=0, bias=False))
        self.drop_out = nn.Dropout(p=0.7)
    def forward(self, x):
        out = self.conv_1(x)
        feat = out.permute(0, 2, 1)
        out = self.drop_out(out)
        cas = self.classifier(out)
        cas = cas.permute(0, 2, 1)
        return feat, cas


class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        config = MainConfig(args)
        #####################################################################
        self.inference_dict = config.inference
        #####################################################################
        self.num_classes = config.dataset["num_classes"] 
        self.r_act = config.dataset["r_act"] 
        self.input_dim = config.dataset["input_dim"]   
        self.len_feature = config.dataset["len_feature"]
        self.max_seq_len =  config.dataset["max_seq_len"]        
        self.feat_stride = config.dataset["feat_stride"] 
        self.num_frames = config.dataset["num_frames"] 
        #####################################################################        
        self.train_center_sample_radius = config.train_cfg["sample_radius"] 
        self.train_cls_prior_prob= config.train_cfg["train_cls_prior_prob"]
        self.head_empty_cls= config.train_cfg["head_empty_cls"]
        #####################################################################
        self.fpn_levels = config.model["fpn_levels"] 
        self.fpn_start_level=config.model["fpn_start_level"]       
        self.fpn_dim = config.model["fpn_dim"] 
        self.fpn_with_ln = config.model["fpn_with_ln"]  
        self.scale_factor =  config.model["scale_factor"] 
        self.n_head = config.model["n_head"] 
        self.n_mha_win_size = config.model["n_mha_win_size"]
        self.embd_kernel_size = config.model["embd_kernel_size"]                      
        self.embd_dim=config.model["embd_dim"]    
        self.embd_with_ln=config.model["embd_with_ln"]       
        self.head_dim = config.model["head_dim"]
        self.head_kernel_size= config.model["head_kernel_size"]
        self.head_num_layers= config.model["head_num_layers"]
        self.head_with_ln= config.model["head_with_ln"]
        #####################################################################        
        self.backbone_arch = (2, 2, self.fpn_levels)
        self.mha_win_size = [self.n_mha_win_size]*(1 + self.fpn_levels)
        self.fpn_strides = [self.scale_factor**i for i in range(self.fpn_start_level, self.fpn_levels+1)]
        self.max_buffer_len_factor = 1 + self.fpn_levels  # defines the max length of the buffered points
        #####################################################################
        all_intervals = []
        start = 0 
        for idx in range(len(self.fpn_strides)):
            level_stride = self.fpn_strides[idx] 
            level_duration = self.max_seq_len/level_stride
            end = int(start+ level_duration)
            all_intervals.append([start,end])
            start = end 
        self.fpn_intervals = all_intervals
        #####################################################################
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.mha_win_size)):
            stride = s * (w // 2) * 2 if w > 1 else s
            assert self.max_seq_len % stride == 0, "max_seq_len must be divisible by fpn stride and window size"
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor
        #####################################################################
        self.cls_module = Cls_Module(self.len_feature, self.num_classes)
        self.criterion = Total_loss(self.r_act)
        self.sigmoid = nn.Sigmoid()
        #####################################################################
        self.backbone = ConvTransformerBackbone(
            **{
                'n_in' : self.input_dim,
                'n_embd' : self.embd_dim,
                'n_head': self.n_head,
                'n_embd_ks': self.embd_kernel_size,
                'max_len': self.max_seq_len,
                'arch' : self.backbone_arch,
                'mha_win_size': self.mha_win_size,
                'scale_factor' : self.scale_factor,
                'with_ln' : self.embd_with_ln,
                'attn_pdrop' : 0.0,
                'proj_pdrop' : 0.0, # dropout ratios for tranformers
                'path_pdrop' : 0.1, # ratio for drop path
                'use_abs_pe' : False, # disable abs position encoding (added to input embedding)    
                'use_rel_pe' : False  # use rel position encoding (added to self-attention)
            })
        #####################################################################
        self.neck = FPNIdentity(
            **{
                'in_channels' : [self.embd_dim] * (self.fpn_levels + 1),
                'out_channel' : self.fpn_dim,
                'scale_factor' : self.scale_factor,
                'start_level' : self.fpn_start_level,
                'with_ln' : self.fpn_with_ln
            })
        #####################################################################
        self.cls_head = PtTransformerClsHead(
            input_dim = self.fpn_dim, feat_dim = self.head_dim, num_classes= self.num_classes+1,
            kernel_size=self.head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=self.head_with_ln,
            num_layers=self.head_num_layers,
            empty_cls=self.head_empty_cls)        
        #####################################################################
        # location generator: points
        self.point_generator = PointGenerator(
            **{
                'max_seq_len' : self.max_seq_len * self.max_buffer_len_factor,
                'fpn_strides' : self.fpn_strides})
        #####################################################################

    @property
    def device(self):
        # a hacky way to get the device type. will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def forward(self, final_dict,is_training):
        # batch the video list into feats (B, C, T) and masks (B, 1, T)
        ###################################################################################
        batched_inputs, batched_masks = self.preprocessing(final_dict)
        ###################################################################################
        feats, masks = self.backbone(batched_inputs, batched_masks)
        fpn_feats, fpn_masks = self.neck(feats, masks)
        ###################################################################################
        out_cls_logits, out_cls_logits_fuse = self.cls_head(fpn_feats, fpn_masks) #[B, T_i, C+1]        
        fpn_masks = [x.squeeze(1) for x in fpn_masks] #list[B, 1, T_i] -> F List[B, T_i]
        ###################################################################################
        if is_training:
            points = self.point_generator(fpn_feats)
            final_points = [x['final_points'].to(self.device) for x in final_dict] #N x 2
            gt_segments = [x['final_proposals'].to(self.device) for x in final_dict] #N x 2
            gt_labels = [x['final_labels'].to(self.device) for x in final_dict] #N x 1
            vid_labels = [x['label_np_arr'].to(self.device) for x in final_dict] #C         
            gt_cls_labels = self.label_points(points, gt_segments, gt_labels, final_points)
            losses = self.losses(fpn_masks, out_cls_logits, out_cls_logits_fuse, gt_cls_labels, vid_labels)
            return losses
        else:
            is_correct, final_res, vid_idxs = self.inference(final_dict, fpn_masks, out_cls_logits ,out_cls_logits_fuse)
            return is_correct, final_res, vid_idxs


    @torch.no_grad()
    def inference(self, video_list, fpn_masks, out_cls_logits, out_cls_logits_fuse):
        vid_idxs = [x['video_id'] for x in video_list][0]
        vid_fps = [x['fps'] for x in video_list][0]
        vid_lens = [x['duration'] for x in video_list][0]
        vid_labels = [x['label_np_arr'] for x in video_list][0]
        vid_labels = vid_labels.cpu().data.numpy()
        ######################################################
        idx = 0 
        cas_sigmoid = [x[idx] for x in out_cls_logits]
        cas_sigmoid_fuse = [x[idx] for x in out_cls_logits_fuse]
        fpn_masks_per_vid = [x[idx] for x in fpn_masks]
        ######################################################        
        class_thresh= self.inference_dict["class_thresh"]
        act_thresh_cas= self.inference_dict["act_thresh_cas"]
        act_thresh_agnostic = self.inference_dict["act_thresh_agnostic"]
        nms_thresh = self.inference_dict["nms_thresh"]
        ######################################################
        fpn_pred_list = []
        proposal_dict = {}
        for fpn_idx in range(len(self.fpn_strides)): 
            temp_scale = self.inference_dict["temp_scale"]*self.fpn_strides[fpn_idx]
            ######################################################
            mask_i= fpn_masks_per_vid[fpn_idx].unsqueeze(-1)
            ######################################################
            cas_sigmoid_fpn = cas_sigmoid[fpn_idx]*mask_i #[T_i, C+1]
            k_act = cas_sigmoid_fpn.shape[0] // self.r_act
            value, _ = cas_sigmoid_fpn.sort(descending=True, dim=0)
            topk_scores = value[:k_act,:-1] 
            vid_score = torch.mean(topk_scores, dim=0) #[C]
            ######################################################
            score_np = vid_score.cpu().data.numpy() 
            pred = np.where(score_np >= class_thresh)[0]
            if len(pred) == 0:
                pred = np.array([np.argmax(score_np)])
            fpn_pred_list += pred.tolist() 
            ######################################################
            cas_sigmoid_fuse_fpn = cas_sigmoid_fuse[fpn_idx] #[T_i, C+1]            
            cls_scores = cas_sigmoid_fuse_fpn[:,:-1]*mask_i  #[T_i, C]
            cas_pred = cls_scores.cpu().numpy()[:, pred]
            cas_pred = main_utils.upgrade_resolution(cas_pred, temp_scale)
            ######################################################
            for thresh_idx in range(len(act_thresh_cas)):
                cas_temp = cas_pred.copy()
                zero_location = np.where(cas_temp[:, :] < act_thresh_cas[thresh_idx])
                cas_temp[zero_location] = 0
                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(cas_temp[:, c] > 0)
                    seg_list.append(pos)                                
                proposals = main_utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, temp_scale, self.fpn_strides[fpn_idx], vid_fps, vid_lens, self.feat_stride, self.num_frames)                                
                for prop_idx in range(len(proposals)):
                    class_id = proposals[prop_idx][0][0]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[prop_idx]
            #####################################################
            agnostic_score = (1 - cas_sigmoid_fuse_fpn[:,-1].unsqueeze(-1))*mask_i #[T_i, 1]
            agnostic_score = agnostic_score.expand((-1, self.num_classes)) #[T_i, C]
            agnostic_score_np = agnostic_score.cpu().data.numpy()[:, pred] #[T_i, pred]
            agnostic_score_np = main_utils.upgrade_resolution(agnostic_score_np, temp_scale)
            ######################################################
            for thresh_idx in range(len(act_thresh_agnostic)):
                cas_temp = cas_pred.copy()
                agnostic_score_np_temp = agnostic_score_np.copy() #[T_i, pred]
                zero_location = np.where(agnostic_score_np_temp[:, :] < act_thresh_agnostic[thresh_idx])
                agnostic_score_np_temp[zero_location] = 0
                seg_list = []
                for c in range(len(pred)):
                    pos = np.where(agnostic_score_np_temp[:, c] > 0)
                    seg_list.append(pos)
                proposals = main_utils.get_proposal_oic(seg_list, cas_temp, score_np, pred, temp_scale,  self.fpn_strides[fpn_idx], vid_fps, vid_lens, self.feat_stride, self.num_frames)
                for prop_idx in range(len(proposals)):
                    class_id = proposals[prop_idx][0][0]
                    if class_id not in proposal_dict.keys():
                        proposal_dict[class_id] = []
                    proposal_dict[class_id] += proposals[prop_idx]
        ##########################################################
        fpn_pred_list = list(set(fpn_pred_list))
        fpn_pred_arr = np.zeros_like(score_np)
        for pred_label in fpn_pred_list:
            fpn_pred_arr[pred_label] =1 
        ##########################################################
        correct_pred = np.sum(vid_labels == fpn_pred_arr, axis=0)
        is_correct= np.sum((correct_pred == self.num_classes).astype(np.float32))
        ##########################################################
        final_proposals = []
        for class_id in proposal_dict.keys():
            final_proposals.append(main_utils.nms(proposal_dict[class_id], thresh=nms_thresh))
        final_proposals = [final_proposals[i][j] for i in range(len(final_proposals)) for j in range(len(final_proposals[i]))]
        final_res= main_utils.result2json(final_proposals)
        ##########################################################
        return is_correct, final_res, vid_idxs

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels, final_points):
        concat_points = torch.cat(points, dim=0) #[sum{T_i},4]
        gt_cls = []        
        for gt_segment, gt_label, final_pt in zip(gt_segments, gt_labels, final_points): # loop over each video sample
            cls_targets = self.label_points_single_video( concat_points, gt_segment, gt_label, final_pt)           
            gt_cls.append(cls_targets)  
        return gt_cls

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label, final_pt):
        # concat_points : F T x 4  -> sum{T_i} x 4 -> (t, range, stride)
        # gt_segment: N x 2 
        # gt_label  : N 
        num_pts = concat_points.shape[0] #sum{T_i}
        num_gts = gt_segment.shape[0] # N
        if num_gts == 0: # corner case where current sample does not have actions
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0) 
            return cls_targets
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        lens = lens[None, :].repeat(num_pts, 1)  
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)  # sum{T_i} x N x 2
        center_pts = final_pt[None].expand(num_pts, num_gts) 
        t_mins =  center_pts - concat_points[:, 1, None] * self.train_center_sample_radius  # sum{T_i} x N 
        t_maxs = center_pts + concat_points[:, 1, None] * self.train_center_sample_radius   # sum{T_i} x N
        cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
        cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
        center_seg = torch.stack((cb_dist_left, cb_dist_right), -1) # sum{T_i} x N x 2
        inside_gt_seg_mask = center_seg.min(-1)[0] > 0 #making sure that the boundaries are positive values! 
        # if there are still more than one actions for one moment, pick the one with the shortest duration 
        lens.masked_fill_(inside_gt_seg_mask==0, float('inf'))
        min_len, min_len_inds = lens.min(dim=1) #min value across 2nd dimension (over N instances)
        # min_len shape -> sum{T_i}
        # corner case: multiple actions with very similar durations (e.g., THUMOS14)
        min_len_mask = torch.logical_and((lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))).to(gt_segment.dtype) 
        # min_len_mask shape -> sum{T_i} x N
        # cls_targets: F T x C
        gt_label_one_hot = F.one_hot(gt_label, self.num_classes).to(gt_segment.dtype) #N x C 
        cls_targets = min_len_mask @ gt_label_one_hot  #sum{T_i} x C 
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        return cls_targets

    def losses(self, fpn_masks, out_cls_logits, out_cls_logits_fuse, gt_cls_labels,vid_labels):
        # out_cls_logits, out_cls_logits_fuse #[B, T_i, C+1]
        ###############################################################################
        vid_labels = torch.stack(vid_labels) # [B , C]
        valid_mask = torch.cat(fpn_masks, dim=1) #[B x sum{T_i}]
        ###############################################################################
        cas_sigmoid = torch.cat(out_cls_logits, dim=1) #[B, sum{T_i}, C+1]
        cas_sigmoid_fuse = torch.cat(out_cls_logits_fuse, dim=1) #[B, sum{T_i}, C+1]
        ###############################################################################
        gt_cls = torch.stack(gt_cls_labels) # [1 x sum{T_i} x C]
        ###############################################################################
        loss_vid, loss_frame, loss_frame_bkg = self.criterion(cas_sigmoid_fuse, cas_sigmoid, valid_mask, gt_cls, vid_labels, self.fpn_intervals)
        final_loss = loss_vid + 0.5 *loss_frame+ loss_frame_bkg
        return {'loss_vid': loss_vid,
                'loss_frame' : loss_frame,
                'loss_bg'    : loss_frame_bkg,
                'final_loss' : final_loss}

    @torch.no_grad()
    def preprocessing(self, final_dict, padding_val=0.0):
        #############################################################################################
        feats = [x['feats'] for x in final_dict]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()
        #############################################################################################
        if self.training:            
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            max_len = self.max_seq_len            
            batch_shape = [len(feats), feats[0].shape[0], max_len] # batch input shape B, C, T
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)
        #############################################################################################
        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]
        #############################################################################################
        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)
        #############################################################################################
        return batched_inputs, batched_masks 
