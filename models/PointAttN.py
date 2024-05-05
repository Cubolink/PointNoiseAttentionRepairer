from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from utils.model_utils import *

from pytorch3d.ops import sample_farthest_points


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1


class GDP(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.cross_transformer = cross_transformer(d_model, d_model_out, nhead, dim_feedforward, dropout)

    def forward(self, x0, points, k_samples):
        """
        Args:
            x0: encoded points
            points:
            k_samples: K farthest point samples

        Returns:

        """
        points, idx_0 = sample_farthest_points(points.transpose(1, 2).contiguous(), K=k_samples)
        x_g0 = gather_points(x0, idx_0)
        points = points.transpose(1, 2)

        x1 = self.cross_transformer(x_g0, x0).contiguous()
        x1 = torch.cat([x_g0, x1], dim=1)

        return x1, points


class SFA(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.cross_transformer = cross_transformer(d_model, d_model_out, nhead, dim_feedforward, dropout)

    def forward(self, x):
        x1 = self.cross_transformer(x, x).contiguous()
        return x1


class PointGenerator(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(PointGenerator, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sfa1 = SFA(channel*2, 512)
        self.sfa2 = SFA(512, 512)
        self.sfa3 = SFA(512, channel*ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel*1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel*ratio, channel*ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

    def forward(self, coarse, feat_g):
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))  # B, C, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))  # B, C, N
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)

        y1 = self.sfa1(y0)
        y2 = self.sfa2(y1)
        y3 = self.sfa3(y2)
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N*self.ratio)

        y_up = y.repeat(1, 1, self.ratio)
        y_cat = torch.cat([y3, y_up], dim=1)
        y4 = self.conv_delta(y_cat)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)

        return x, y3


class FeatureExtractor(nn.Module):
    def __init__(self, channel=64):
        super(FeatureExtractor, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.gdp1 = GDP(channel, channel)
        self.sfa1 = SFA(channel*2, channel*2)
        self.gdp2 = GDP(channel*2, channel*2)
        self.sfa2 = SFA(channel*4, channel*4)
        self.gdp3 = GDP(channel*4, channel*4)
        self.sfa3 = SFA(channel*8, channel*8)

        self.relu = nn.GELU()

    def forward(self, points):
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, D, N
        x0 = self.conv2(x)

        # # Feature Extractor
        # GDP
        x1, points = self.gdp1(x0, points, N//4)
        # SFA
        x1 = self.sfa1(x1).contiguous()
        # GDP
        x2, points = self.gdp2(x1, points, N//8)
        # SFA
        x2 = self.sfa2(x2)
        # GDP
        x3, _ = self.gdp3(x2, points, N//16)
        # SFA
        x3 = self.sfa3(x3)
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)

        return x_g


class SeedGenerator(nn.Module):
    def __init__(self, channel=64):
        super(SeedGenerator, self).__init__()
        self.channel = channel

        self.sfa0_d = SFA(channel * 8, channel * 8)
        self.sfa1_d = SFA(channel * 8, channel * 8)
        self.sfa2_d = SFA(channel * 8, channel * 8)

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel * 4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel * 8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel * 8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel * 8, channel * 8, kernel_size=1)

        self.relu = nn.GELU()

    def forward(self, x_g, points):
        batch_size, _, N = points.size()

        # # Seed Generator
        x = self.relu(self.ps_adj(x_g))
        x = self.relu(self.ps(x))
        x = self.relu(self.ps_refuse(x))
        # SFA
        x0_d = (self.sfa0_d(x))
        x1_d = (self.sfa1_d(x0_d))
        x2_d = (self.sfa2_d(x1_d)).reshape(batch_size, self.channel * 4, N // 8)

        coarse = self.conv_out(self.relu(self.conv_out1(x2_d)))

        new_x = torch.cat([points, coarse], dim=2)
        new_x, _ = sample_farthest_points(new_x.transpose(1, 2).contiguous(), K=512)
        new_x = new_x.transpose(1, 2)

        return new_x, coarse


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif args.dataset == 'c3d':
            step1 = 1
            step2 = 4
        elif args.dataset == 'chs':
            step1 = 4
            step2 = 8
        else:
            raise ValueError('dataset does not exist')

        self.feature_extractor = FeatureExtractor()
        self.seed_generator = SeedGenerator()

        self.refine = PointGenerator(ratio=step1)
        self.refine1 = PointGenerator(ratio=step2)

    def forward(self, x, gt=None, is_training=True):
        feat_g = self.feature_extractor(x)
        seeds, coarse = self.seed_generator(feat_g, x)

        fine, _ = self.refine(seeds, feat_g)
        fine1, _ = self.refine1(fine, feat_g)

        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()

        if is_training:
            loss3, _ = calc_cd(fine1, gt)
            gt_fine1, _ = sample_farthest_points(gt, K=fine.shape[1])

            loss2, _ = calc_cd(fine, gt_fine1)
            gt_coarse, _ = sample_farthest_points(gt_fine1, K=coarse.shape[1])

            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return fine, loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

            return {
                'out1': coarse, 'out2': fine1,
                'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse,
                'cd_p': cd_p, 'cd_t': cd_t
            }
