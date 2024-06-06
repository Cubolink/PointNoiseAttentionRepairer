from .PointAttN import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset != 'chs':
            raise ValueError('dataset does not exist')

        self.feature_extractor = FeatureExtractor()
        self.seed_generator = SeedGenerator()
        self.refine = Denoiser()

    def forward(self, x, noise, gt_coarse=None, gt=None, is_training=True):
        if gt_coarse is None:
            gt_coarse = gt

        feat_g = self.feature_extractor(x)
        seeds, coarse = self.seed_generator(feat_g, x)
        unnoised = self.refine(seeds, feat_g, noise)

        coarse = coarse.transpose(1, 2).contiguous()
        unnoised = unnoised.transpose(1, 2).contiguous()

        if is_training:
            gt, _ = sample_farthest_points(gt, K=unnoised.shape[1])
            loss3, _ = calc_cd(unnoised, gt)

            gt_coarse, _ = sample_farthest_points(gt_coarse, K=coarse.shape[1])
            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss3.mean()
            return loss3, loss1, total_train_loss
        else:
            cd_p, cd_t = calc_cd(unnoised, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt_coarse)

            return {
                'out1': coarse, 'out2': unnoised,
                'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse,
                'cd_p': cd_p, 'cd_t': cd_t
            }
