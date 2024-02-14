# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import sys
sys.path.append(['../..'])
import os
import os.path as osp
import numpy as np
import pickle
import logging

from sklearn.model_selection import train_test_split
import argparse
import random

import sys
import traceback
import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))



def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_masked_input_and_labels(inp, mask_value=1, mask_p=0.15, mask_random_p=0.1, mask_remain_p=0.1, mask_random_s=1):
    # BERT masking
    inp_mask = (torch.rand(*inp.shape[:2]) < mask_p).to(inp.device)

    # Prepare input
    inp_masked = inp.clone().float()

    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = (inp_mask & (torch.rand(*inp.shape[:2]) < 1 - mask_remain_p).to(inp.device))
    inp_masked[inp_mask_2mask] = mask_value # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (torch.rand(*inp.shape[:2]) < mask_random_p / (1 - mask_remain_p)).to(inp.device)

    inp_masked[inp_mask_2random] = (2 * mask_random_s * torch.rand(inp_mask_2random.sum().item(), inp.shape[2]) - mask_random_s).to(inp.device)

    # y_labels would be same as encoded_texts i.e input tokens
    gt = inp.clone()
    return inp_masked, gt

def random_rot_mat(bs, uniform_dist):
    rot_mat = torch.zeros(bs, 3, 3)
    random_values = uniform_dist.rsample((bs,))
    rot_mat[:, 0, 0] = torch.cos(random_values)
    rot_mat[:, 0, 1] = -torch.sin(random_values)
    rot_mat[:, 1, 0] = torch.sin(random_values)
    rot_mat[:, 1, 1] = torch.cos(random_values)
    rot_mat[:, 2, 2] = 1
    return rot_mat

def repeat_rot_mat(rot_mat, num):
    batch = rot_mat.shape[0]
    res = torch.zeros([batch, 3*num, 3*num]).to(rot_mat.device)
    for i in range(num):
        res[:, 3*i:3*(i+1), 3*i:3*(i+1)] = rot_mat
    return res

def align_skeleton(data):
    N, C, T, V, M = data.shape
    trans_data = np.zeros_like(data)
    for i in tqdm(range(N)):
        for p in range(M):
            sample = data[i][..., p]
            # if np.all((sample[:,0,:] == 0)):
                # continue
            d = sample[:,0,1:2]
            v1 = sample[:,0,1]-sample[:,0,0]
            if np.linalg.norm(v1) <= 0.0:
                continue
            v1 = v1/np.linalg.norm(v1)
            v2_ = sample[:,0,12]-sample[:,0,16]
            proj_v2_v1 = np.dot(v1.T,v2_)*v1/np.linalg.norm(v1)
            v2 = v2_-np.squeeze(proj_v2_v1)
            v2 = v2/(np.linalg.norm(v2))
            v3 = np.cross(v2,v1)/(np.linalg.norm(np.cross(v2,v1)))
            v1 = np.reshape(v1,(3,1))
            v2 = np.reshape(v2,(3,1))
            v3 = np.reshape(v3,(3,1))

            R = np.hstack([v2,v3,v1])
            for t in range(T):
                trans_sample = (np.linalg.inv(R))@(sample[:,t,:]) # -d
                trans_data[i, :, t, :, p] = trans_sample
    return trans_data

def create_aligned_dataset(file_list=['data/ntu/NTU60_CS.npz', 'data/ntu/NTU60_CV.npz']):
    for file in file_list:
        org_data = np.load(file)
        splits = ['x_train', 'x_test']
        aligned_set = {}
        for split in splits:
            data = org_data[split]
            N, T, _ = data.shape
            data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
            aligned_data = align_skeleton(data)
            aligned_data = aligned_data.transpose(0, 2, 4, 3, 1).reshape(N, T, -1)
            aligned_set[split] = aligned_data

        np.savez(file.replace('.npz', '_aligned.npz'),
                 x_train=aligned_set['x_train'],
                 y_train=org_data['y_train'],
                 x_test=aligned_set['x_test'],
                 y_test=org_data['y_test'])



def get_motion(data, data_format=['x'], use_nonzero_mask=False, rot=False, jittering=False, random_dist=None):
    N, C, T, V, M = data.size()
    data = data.permute(0, 4, 2, 3, 1).contiguous().view(N*M, T, V, C)

    # get motion features
    x = data - data[:,:,0:1,:] # localize
    if 'v' in data_format:
        v = x[:,1:,:,:] - x[:,:-1,:,:]
        v = torch.cat([torch.zeros(N*M, 1, V, C).to(v.device), v], dim=1)
    if 'a' in data_format:
        a = v[:,1:,:,:] - v[:,:-1,:,:]
        a = torch.cat([torch.zeros(N*M, 1, V, C).to(a.device), a], dim=1)

    # reshape x,v for PORT
    x = x.view(N*M*T, V, C)
    if 'v' in data_format:
        v = v.view(N*M*T, V, C)
    if 'a' in data_format:
        a = a.view(N*M*T, V, C)

    # apply nonzero mask
    if use_nonzero_mask:
        nonzero_mask = x.view(N*M*T, -1).count_nonzero(dim=-1) !=0
        x = x[nonzero_mask]
        if 'v' in data_format:
            v = v[nonzero_mask]
        if 'a' in data_format:
            a = a[nonzero_mask]

    # optionally rotate
    if rot:
        rot_mat = random_rot_mat(x.shape[0], random_dist).to(x.device)
        x = x.transpose(1, 2) # (NMT, C, V)
        x = torch.bmm(rot_mat, x) # rotate
        x = x.transpose(1, 2) #(NMT, V, C)

        if 'v' in data_format:
            v = v.transpose(1, 2) # (NMT, C, V)
            v = torch.bmm(rot_mat, v) # rotate
            v = v.transpose(1, 2) #(NMT, V, C)

        if 'a' in data_format:
            a = a.transpose(1, 2) # (NMT, C, V)
            a = torch.bmm(rot_mat, a) # rotate
            a = a.transpose(1, 2) #(NMT, V, C)

    if jittering:
        jit = (torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) - 0.5) / 10
        x += jit

    output = {'x':x}
    if 'v' in data_format:
        output['v'] = v
    if 'a' in data_format:
        output['a'] = a

    return output

def get_attn(x, mask= None, similarity='scaled_dot'):
    if similarity == 'scaled_dot':
        sqrt_dim = np.sqrt(x.shape[-1])
        score = torch.bmm(x, x.transpose(1, 2)) / sqrt_dim
    elif similarity == 'euclidean':
        score = torch.cdist(x, x)

    if mask is not None:
        score.masked_fill_(mask.view(score.size()), -float('Inf'))

    attn = F.softmax(score, -1)
    embd = torch.bmm(attn, x)
    return embd, attn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def padding_preprocess(root_path, target_path):

    stat_path = osp.join(root_path, 'statistics')
    setup_file = osp.join(stat_path, 'setup.txt')
    camera_file = osp.join(stat_path, 'camera.txt')
    performer_file = osp.join(stat_path, 'performer.txt')
    replication_file = osp.join(stat_path, 'replication.txt')
    label_file = osp.join(stat_path, 'label.txt')
    skes_name_file = osp.join(stat_path, 'skes_available_name.txt')

    denoised_path = osp.join(root_path, 'denoised_data')
    raw_skes_joints_pkl = osp.join(denoised_path, 'raw_denoised_joints.pkl')
    frames_file = osp.join(denoised_path, 'frames_cnt.txt')

    save_path = target_path


    if not osp.exists(save_path):
        os.mkdir(save_path)


    def remove_nan_frames(ske_name, ske_joints, nan_logger):
        num_frames = ske_joints.shape[0]
        valid_frames = []

        for f in range(num_frames):
            if not np.any(np.isnan(ske_joints[f])):
                valid_frames.append(f)
            else:
                nan_indices = np.where(np.isnan(ske_joints[f]))[0]
                nan_logger.info('{}\t{:^5}\t{}'.format(ske_name, f + 1, nan_indices))

        return ske_joints[valid_frames]

    def seq_translation(skes_joints):
        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            num_bodies = 1 if ske_joints.shape[1] == 75 else 2
            if num_bodies == 2:
                missing_frames_1 = np.where(ske_joints[:, :75].sum(axis=1) == 0)[0]
                missing_frames_2 = np.where(ske_joints[:, 75:].sum(axis=1) == 0)[0]
                cnt1 = len(missing_frames_1)
                cnt2 = len(missing_frames_2)

            i = 0  # get the "real" first frame of actor1
            while i < num_frames:
                if np.any(ske_joints[i, :75] != 0):
                    break
                i += 1

            origin = np.copy(ske_joints[i, 3:6])  # new origin: joint-2

            for f in range(num_frames):
                if num_bodies == 1:
                    ske_joints[f] -= np.tile(origin, 25)
                else:  # for 2 actors
                    ske_joints[f] -= np.tile(origin, 50)

            if (num_bodies == 2) and (cnt1 > 0):
                ske_joints[missing_frames_1, :75] = np.zeros((cnt1, 75), dtype=np.float32)

            if (num_bodies == 2) and (cnt2 > 0):
                ske_joints[missing_frames_2, 75:] = np.zeros((cnt2, 75), dtype=np.float32)

            skes_joints[idx] = ske_joints  # Update

        return skes_joints


    def frame_translation(skes_joints, skes_name, frames_cnt):
        nan_logger = logging.getLogger('nan_skes')
        nan_logger.setLevel(logging.INFO)
        nan_logger.addHandler(logging.FileHandler("./nan_frames.log"))
        nan_logger.info('{}\t{}\t{}'.format('Skeleton', 'Frame', 'Joints'))

        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            # Calculate the distance between spine base (joint-1) and spine (joint-21)
            j1 = ske_joints[:, 0:3]
            j21 = ske_joints[:, 60:63]
            dist = np.sqrt(((j1 - j21) ** 2).sum(axis=1))

            for f in range(num_frames):
                origin = ske_joints[f, 3:6]  # new origin: middle of the spine (joint-2)
                if (ske_joints[f, 75:] == 0).all():
                    ske_joints[f, :75] = (ske_joints[f, :75] - np.tile(origin, 25)) / \
                                        dist[f] + np.tile(origin, 25)
                else:
                    ske_joints[f] = (ske_joints[f] - np.tile(origin, 50)) / \
                                    dist[f] + np.tile(origin, 50)

            ske_name = skes_name[idx]
            ske_joints = remove_nan_frames(ske_name, ske_joints, nan_logger)
            frames_cnt[idx] = num_frames  # update valid number of frames
            skes_joints[idx] = ske_joints

        return skes_joints, frames_cnt


    def align_frames(skes_joints, frames_cnt):
        """
        Align all sequences with the same frame length.

        """
        num_skes = len(skes_joints)
        max_num_frames = frames_cnt.max()  # 300
        aligned_skes_joints = np.zeros((num_skes, max_num_frames, 150), dtype=np.float32)

        for idx, ske_joints in enumerate(skes_joints):
            num_frames = ske_joints.shape[0]
            num_bodies = 1 if ske_joints.shape[1] == 75 else 2
            if num_bodies == 1:
                aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, ske_joints))
                # aligned_skes_joints[idx, :num_frames] = np.hstack((ske_joints, np.zeros_like(ske_joints)))
            else:
                aligned_skes_joints[idx, :num_frames] = ske_joints

        return aligned_skes_joints


    def one_hot_vector(labels):
        num_skes = len(labels)
        labels_vector = np.zeros((num_skes, 120))
        for idx, l in enumerate(labels):
            labels_vector[idx, l] = 1

        return labels_vector


    def split_train_val(train_indices, method='sklearn', ratio=0.05):
        """
        Get validation set by splitting data randomly from training set with two methods.
        In fact, I thought these two methods are equal as they got the same performance.

        """
        if method == 'sklearn':
            return train_test_split(train_indices, test_size=ratio, random_state=10000)
        else:
            np.random.seed(10000)
            np.random.shuffle(train_indices)
            val_num_skes = int(np.ceil(0.05 * len(train_indices)))
            val_indices = train_indices[:val_num_skes]
            train_indices = train_indices[val_num_skes:]
            return train_indices, val_indices


    def split_dataset(skes_joints, label, performer, setup, evaluation, save_path):
        train_indices, test_indices = get_indices(performer, setup, evaluation)

        train_labels = label[train_indices]
        test_labels = label[test_indices]

        train_x = skes_joints[train_indices]
        train_y = one_hot_vector(train_labels)
        test_x = skes_joints[test_indices]
        test_y = one_hot_vector(test_labels)

        save_name = 'NTU120_%s.npz' % evaluation
        np.savez(save_name, x_train=train_x, y_train=train_y, x_test=test_x, y_test=test_y)

    

    def get_indices(performer, setup, evaluation='CSub'):
        test_indices = np.empty(0)
        train_indices = np.empty(0)

        if evaluation == 'CSub':  # Cross Subject (Subject IDs)
            train_ids = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28,
                        31, 34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57,
                        58, 59, 70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92,
                        93, 94, 95, 97, 98, 100, 103]
            test_ids = [i for i in range(1, 107) if i not in train_ids]

            # Get indices of test data
            for idx in test_ids:
                temp = np.where(performer == idx)[0]  # 0-based index
                test_indices = np.hstack((test_indices, temp)).astype(np.int32)

            # Get indices of training data
            for train_id in train_ids:
                temp = np.where(performer == train_id)[0]  # 0-based index
                train_indices = np.hstack((train_indices, temp)).astype(np.int32)
        else:  # Cross Setup (Setup IDs)
            train_ids = [i for i in range(1, 33) if i % 2 == 0]  # Even setup
            test_ids = [i for i in range(1, 33) if i % 2 == 1]  # Odd setup

            # Get indices of test data
            for test_id in test_ids:
                temp = np.where(setup == test_id)[0]  # 0-based index
                test_indices = np.hstack((test_indices, temp)).astype(np.int32)

            # Get indices of training data
            for train_id in train_ids:
                temp = np.where(setup == train_id)[0]  # 0-based index
                train_indices = np.hstack((train_indices, temp)).astype(np.int32)

        return train_indices, test_indices

    setup = np.loadtxt(setup_file, dtype=np.int32)  # camera id: 1~32
    performer = np.loadtxt(performer_file, dtype=np.int32)  # subject id: 1~106
    label = np.loadtxt(label_file, dtype=np.int32) - 1  # action label: 0~119

    frames_cnt = np.loadtxt(frames_file, dtype=np.int32)  # frames_cnt
    skes_name = np.loadtxt(skes_name_file, dtype=np.string_)

    with open(raw_skes_joints_pkl, 'rb') as fr:
        skes_joints = pickle.load(fr)  # a list

    skes_joints = seq_translation(skes_joints)
    skes_joints = align_frames(skes_joints, frames_cnt)  

    evaluations = ['CSet', 'CSub']
    for evaluation in evaluations:
        split_dataset(skes_joints, label, performer, setup, evaluation, save_path)
    create_aligned_dataset(file_list=['NTU120_CSet.npz', 'NTU120_Csub.npz'])
