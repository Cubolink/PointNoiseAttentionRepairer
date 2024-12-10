from .PointAttN import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        if args.dataset != 'chs':
            raise ValueError('dataset does not exist')

        self.feature_extractor = FeatureExtractor()
        self.seed_generator = SeedGenerator()
        self.refine = PrimalExtractor()

    @staticmethod
    def _filter_noise(noise, mask):
        filtered = torch.zeros(noise.shape, device=noise.device)
        sizes = mask.sum(axis=1)
        pad_sizes = 2048 - sizes
        for b in range(len(filtered)):
            filtered[b] = nn.functional.pad(noise[b, mask[b], :], (0, 0, 0, pad_sizes[b]), "constant", 42)
        return filtered

    def forward(self, x, noise, gt_coarse=None, gt=None, is_training=True):
        gt_is_occ = (len(gt.shape) == 2) if gt is not None else False  # should be true on training

        feat_g = self.feature_extractor(x)
        seeds, coarse = self.seed_generator(feat_g, x)
        logits, occ_mask = self.refine(seeds, feat_g, noise)

        coarse = coarse.transpose(1, 2).contiguous()
        noise = noise.transpose(1, 2).contiguous()
        logits = torch.squeeze(logits, dim=1)

        occ_mask = occ_mask.squeeze(1)
        filtered_list = self._filter_noise(noise, occ_mask)
        filtered_list_gt = self._filter_noise(noise, gt.bool()) if gt_is_occ else None

        if is_training:
            loss3 = nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            loss3 = loss3.mean(axis=1)

            loss2, _ = calc_cd(filtered_list, filtered_list_gt)

            gt_coarse, _ = sample_farthest_points(gt_coarse, K=coarse.shape[1])
            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return loss3, loss2, loss1, total_train_loss
        else:
            if gt is None:
                return {
                    'out1': coarse,
                    'out2': filtered_list,
                    'occ': torch.sigmoid(logits)
                }
            if gt_is_occ:
                bce = nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(axis=1)
                cd_p, cd_t = calc_cd(filtered_list, filtered_list_gt)
            else:
                bce = None
                cd_p, cd_t = calc_cd(filtered_list, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt_coarse)

            return {
                'out1': coarse,  # predicted missing part
                'out2': filtered_list,  # predicted output
                'occ': torch.sigmoid(logits),  # predicted occupancy probability
                'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse,
                'cd_t': cd_t, 'cd_p': cd_p,
                'bce': bce
            }


def train_step(data, net, do_summary_string):
    _, inputs, noise, gt, restoration_gt = data

    inputs = inputs.float().cuda()
    noise = noise.float().cuda()
    gt = gt.float().cuda()

    inputs = inputs.transpose(2, 1).contiguous()
    noise = noise.transpose(2, 1).contiguous()

    loss1, loss2, loss3, net_loss = net(inputs, noise, gt_coarse=restoration_gt, gt=gt)

    summary_string = None
    if do_summary_string:
        summary_string = (
            f' bce_loss: {loss1.mean().item()}'
            f' fine_loss: {loss2.mean().item()}'
            f' coarse_loss: {loss3.mean().item()}'
            f' total_loss: {net_loss.mean().item()}'
        )

    return net_loss, summary_string


def val_step(data, net):
    label, inputs, noise, gt, restoration_gt = data

    inputs = inputs.float().cuda()
    noise = noise.float().cuda()
    gt = gt.float().cuda()

    inputs = inputs.transpose(2, 1).contiguous()
    noise = noise.transpose(2, 1).contiguous()

    result_dict = net(inputs, noise, gt_coarse=restoration_gt, gt=gt, is_training=False)
    return result_dict
