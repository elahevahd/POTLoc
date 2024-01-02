import os
import time
import numpy as np
import random
import math
import warnings
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import _LRScheduler
from scipy.interpolate import interp1d
####################################################################################
from main_backbones import  MaskedConv1D, Scale, LayerNorm, AffineDropPath
import config
####################################################################################

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Sets the learning rate of each parameter group to follow a linear warmup schedule
    between warmup_start_lr and base_lr followed by a cosine annealing schedule between
    base_lr and eta_min.
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr = 0.0,
        eta_min = 1e-8,
        last_epoch = -1,
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super(LinearWarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute learning rate using chainable form of the scheduler
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        elif (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs))) /
            (
                1 +
                math.cos(math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs))
            ) * (group["lr"] - self.eta_min) + self.eta_min for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Called when epoch is passed as a param to the `step` function of the scheduler.
        """
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


def fix_random_seed(seed, include_cuda=True):
    rng_generator = torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if include_cuda:
        # training: disable cudnn benchmark to ensure the reproducibility
        cudnn.enabled = True
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # this is needed for CUDA >= 10.2
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # torch.use_deterministic_algorithms(True, warn_only=True)
        # torch.use_deterministic_algorithms(True)
    else:
        cudnn.enabled = True
        cudnn.benchmark = True
    return rng_generator

def trivial_batch_collator(batch):
    """
        A batch collator that does nothing
    """
    return batch

def worker_init_reset_seed(worker_id):
    """
        Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2 ** 31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_checkpoint(state, is_best, file_folder,
                    file_name='checkpoint.pth.tar'):
    """save checkpoint to file"""
    if not os.path.exists(file_folder):
        os.mkdir(file_folder)
    torch.save(state, os.path.join(file_folder, file_name))
    if is_best:
        # skip the optimization / scheduler state
        state.pop('optimizer', None)
        state.pop('scheduler', None)
        torch.save(state, os.path.join(file_folder, 'model_best.pth.tar'))


def print_model_params(model):
    for name, param in model.named_parameters():
        print(name, param.min().item(), param.max().item(), param.mean().item())
    return


def make_optimizer(model, optimizer_config):
    """create optimizer
    return a supported optimizer
    """
    # separate out all parameters that with / without weight decay
    # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, MaskedConv1D)
    blacklist_weight_modules = (LayerNorm, torch.nn.GroupNorm)

    # loop over all modules / params
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('scale') and isinstance(m, (Scale, AffineDropPath)):
                # corner case of our scale layer
                no_decay.add(fpn)
            elif pn.endswith('rel_pe'):
                # corner case for relative position encoding
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, \
        "parameters %s were not separated into either decay/no_decay set!" \
        % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": optimizer_config['weight_decay']},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if optimizer_config["type"] == "SGD":
        optimizer = optim.SGD(
            optim_groups,
            lr=optimizer_config["learning_rate"],
            momentum=optimizer_config["momentum"]
        )
    elif optimizer_config["type"] == "AdamW":
        optimizer = optim.AdamW(
            optim_groups,
            lr=optimizer_config["learning_rate"]
        )
    else:
        raise TypeError("Unsupported optimizer!")

    return optimizer


def make_scheduler(
    optimizer,
    optimizer_config,
    num_iters_per_epoch,
    last_epoch=-1
):
    """create scheduler
    return a supported scheduler
    All scheduler returned by this function should step every iteration
    """
    if optimizer_config["warmup"]:
        max_epochs = optimizer_config["epochs"] + optimizer_config["warmup_epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # get warmup params
        warmup_epochs = optimizer_config["warmup_epochs"]
        warmup_steps = warmup_epochs * num_iters_per_epoch

        # with linear warmup: call our custom schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # Cosine
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_steps,
                max_steps,
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    else:
        max_epochs = optimizer_config["epochs"]
        max_steps = max_epochs * num_iters_per_epoch

        # without warmup: call default schedulers
        if optimizer_config["schedule_type"] == "cosine":
            # step per iteration
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                max_steps,
                last_epoch=last_epoch
            )
        else:
            raise TypeError("Unsupported scheduler!")

    return scheduler


class AverageMeter(object):
    """Computes and stores the average and current value.
    Used to compute dataset stats from mini-batches
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0.0

    def initialize(self, val, n):
        self.val = val
        self.avg = val
        self.sum = val * n
        self.count = n
        self.initialized = True

    def update(self, val, n=1):
        if not self.initialized:
            self.initialize(val, n)
        else:
            self.add(val, n)

    def add(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    model_ema = None,
    clip_grad_l2norm = -1,
    tb_writer = None,
    print_freq = 20
):
    """Training the model for one epoch"""
    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {}
    # number of iterations per epoch
    num_iters = len(train_loader)
    # switch to train mode
    model.train()

    # main training loop
    print("\n[Train]: Epoch {:d} started".format(curr_epoch))
    start = time.time()
    for iter_idx, video_list in enumerate(train_loader, 0):
        # zero out optim
        optimizer.zero_grad(set_to_none=True)
        # forward / backward the model
        losses = model(video_list, is_training=True)
        losses['final_loss'].backward()
        # gradient cliping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                clip_grad_l2norm
            )
        # step optimizer / scheduler
        optimizer.step()
        scheduler.step()

        if model_ema is not None:
            model_ema.update(model)

        # printing (only check the stats when necessary to avoid extra cost)
        if (iter_idx != 0) and (iter_idx % print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # track all losses
            for key, value in losses.items():
                # init meter if necessary
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                # update
                losses_tracker[key].update(value.item())

            # log to tensor board
            lr = scheduler.get_last_lr()[0]
            global_step = curr_epoch * num_iters + iter_idx
            if tb_writer is not None:
                # learning rate (after stepping)
                tb_writer.add_scalar(
                    'train/learning_rate',
                    lr,
                    global_step
                )
                # all losses
                tag_dict = {}
                for key, value in losses_tracker.items():
                    if key != "final_loss":
                        tag_dict[key] = value.val
                tb_writer.add_scalars(
                    'train/all_losses',
                    tag_dict,
                    global_step
                )
                # final loss
                tb_writer.add_scalar(
                    'train/final_loss',
                    losses_tracker['final_loss'].val,
                    global_step
                )

            # print to terminal
            block1 = 'Epoch: [{:03d}][{:05d}/{:05d}]'.format(
                curr_epoch, iter_idx, num_iters
            )
            block2 = 'Time {:.2f} ({:.2f})'.format(
                batch_time.val, batch_time.avg
            )
            block3 = 'Loss {:.2f} ({:.2f})\n'.format(
                losses_tracker['final_loss'].val,
                losses_tracker['final_loss'].avg
            )
            block4 = ''
            for key, value in losses_tracker.items():
                if key != "final_loss":
                    block4  += '\t{:s} {:.2f} ({:.2f})'.format(
                        key, value.val, value.avg
                    )

            print('\t'.join([block1, block2, block3, block4]))

    # finish up and print
    lr = scheduler.get_last_lr()[0]
    print("[Train]: Epoch {:d} finished with lr={:.8f}\n".format(curr_epoch, lr))
    return


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0])
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def get_proposal_oic(seg_list, cas_temp, score_np, c_pred, temp_scale, fpn_scale,  vid_fps, vid_lens, feat_stride, num_frames, _lambda=0.2, gamma=0.0):
    temp = []
    for i in range(len(seg_list)):
        c_temp = []
        temp_list = np.array(seg_list[i])[0]
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(cas_temp[grouped_temp_list[j], i])
                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(cas_temp.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))
                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(cas_temp[outer_temp_list, i])
                c_score = inner_score - outer_score + gamma * score_np[c_pred[i]]
                ###########################################################
                delta = (num_frames/2)
                start_index = grouped_temp_list[j][0]
                end_index = (grouped_temp_list[j][-1] + 1) 
                start_index = (start_index / temp_scale)*fpn_scale
                end_index = (end_index / temp_scale)*fpn_scale
                t_start = (start_index * feat_stride + delta)/ vid_fps  
                t_end = (end_index * feat_stride + delta)/vid_fps 
                t_end = min(t_end, vid_lens)
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result):
    result_file = []    
    for i in range(len(result)):
        line = {'label': config.class_dict[result[i][0]], 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file

def save_best_record_thumos(test_info, file_path):
    fo = open(file_path, "w")
    # fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("Test_acc: {:.4f}\n".format(test_info["test_acc"][-1]))
    fo.write("average_mAP[0.1:0.7]: {:.4f}\n".format(test_info["average_mAP[0.1:0.7]"][-1]))
    fo.write("average_mAP[0.1:0.5]: {:.4f}\n".format(test_info["average_mAP[0.1:0.5]"][-1]))
    fo.write("average_mAP[0.3:0.7]: {:.4f}\n".format(test_info["average_mAP[0.3:0.7]"][-1]))    
    tIoU_thresh = np.linspace(0.1, 0.7, 7)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.1f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.1f}".format(tIoU_thresh[i])][-1]))
    fo.close()

def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])
    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta.detach()
    ret[ret > 1] = 1
    ret[ret < 0] = 0
    return ret


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


def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()


def feature_sampling(features, start, end, num_divide):
    step = (end - start) / num_divide
    feature_lst = torch.zeros((num_divide, features.shape[1])).cuda()
    for i in range(num_divide):
        start_point = int(start + step * i)
        end_point = int(start + step * (i+1))        
        if start_point >= end_point:
            end_point += 1
        sample_id = np.random.randint(start_point, end_point)
        feature_lst[i] = features[sample_id]
    return feature_lst.mean(dim=0)


def get_oic_score(cas_sigmoid_fuse, start, end, delta=0.25):
    length = end - start + 1
    inner_score = torch.mean(cas_sigmoid_fuse[start:end+1])
    outer_s = max(0, int(start - delta * length))
    outer_e = min(int(cas_sigmoid_fuse.shape[0] - 1), int(end + delta * length))
    outer_seg = list(range(outer_s, start)) + list(range(end + 1, outer_e + 1))
    if len(outer_seg) == 0:
        outer_score = 0
    else:
        outer_score = torch.mean(cas_sigmoid_fuse[outer_seg])
    return inner_score - outer_score



def select_seed(cas_sigmoid_fuse, point_anno,bkg_thresh = 0.95):  
    #cas_sigmoid_fuse: [B , T_i, C+1], point_anno: [B , T_i, C+1]
    
    point_anno_agnostic = point_anno.max(dim=2)[0]   #[B , T_i]
    bkg_seed = torch.zeros_like(point_anno_agnostic) #[B , T_i]
    bkg_score = cas_sigmoid_fuse[:,:,-1]             #[B , T_i]

    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1) #1D # act_idx = {t_0, t_1, ..., t_k}        
        act_numbers = act_idx.shape[0]

        if act_numbers>0:
            """ most left """
            if act_idx[0] > 0:  #t_0 > 0 
                bkg_score_tmp = bkg_score[b,:act_idx[0]]  #bg_score on [0, t_0)
                idx_tmp = bkg_seed[b,:act_idx[0]]         #bkg_seed on [0, t_0)
                idx_tmp[bkg_score_tmp >= bkg_thresh] = 1
                if idx_tmp.sum() >= 2:
                    start_index = idx_tmp.nonzero().squeeze(1)[0]
                    end_index = idx_tmp.nonzero().squeeze(1)[-1]
                    idx_tmp[start_index+1:end_index] = 1                                   
            """ most right """
            if act_idx[-1] < (point_anno.shape[1] - 1):       #t_k < T_i-1
                bkg_score_tmp = bkg_score[b,act_idx[-1]+1:]   #bg_score on [t_k+1, T_i)
                idx_tmp = bkg_seed[b,act_idx[-1]+1:]          #bkg_seed on [t_k+1, T_i)
                idx_tmp[bkg_score_tmp >= bkg_thresh] = 1
                if idx_tmp.sum() >= 2:
                    start_index = idx_tmp.nonzero().squeeze(1)[0]
                    end_index = idx_tmp.nonzero().squeeze(1)[-1]
                    idx_tmp[start_index+1:end_index] = 1                                   
                
            """ between two instances """
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

    return  bkg_seed