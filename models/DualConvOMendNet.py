from __future__ import print_function
from utils.model_utils import *

from .PointAttN import GDP2
from models.convonet.decoder import LocalDecoder
from models.convonet.encoder.pointnet import LocalPoolPointnet


class ConvONet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim = 3  # point data dimension
        c_dim = cfg['c_dim']
        decoder_kwargs = cfg['decoder_kwargs']
        encoder_kwargs = cfg['encoder_kwargs']
        padding = 0.1

        # local positional encoding
        if 'local_coord' in cfg.keys():
            encoder_kwargs['local_coord'] = cfg['local_coord']
            decoder_kwargs['local_coord'] = cfg['local_coord']
        if 'pos_encoding' in cfg:
            encoder_kwargs['pos_encoding'] = cfg['pos_encoding']
            decoder_kwargs['pos_encoding'] = cfg['pos_encoding']

        self.encoder = LocalPoolPointnet(dim=dim, c_dim=c_dim, padding=padding, **encoder_kwargs)
        self.decoder = LocalDecoder(dim=dim, c_dim=c_dim, padding=padding, **decoder_kwargs)

    def forward(self, p, inputs, sample=True):
        c = self.encoder(inputs)
        kwargs = {'skip_last_layer': True}
        logits = self.decoder(p, c, **kwargs)
        # p_r = torch.distributions.Bernoulli(logits=logits)
        return logits


class AttentionMLP(nn.Module):
    def __init__(self, model_dim, out_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.out_dim = out_dim

        # increased in_dim due to concatenation
        mlp_in_dim = model_dim + model_dim

        self.multihead_attention = nn.MultiheadAttention(model_dim, num_heads)
        self.layer_norm1 = nn.LayerNorm(model_dim)

        self.pointwise_feed = nn.Sequential(nn.Linear(model_dim, model_dim),
                                            nn.ReLU(),
                                            nn.Linear(model_dim, model_dim))
        self.layer_norm2 = nn.LayerNorm(model_dim)

        self.mlp = nn.Sequential(nn.Linear(mlp_in_dim, out_dim),
                                 nn.ReLU(),
                                 nn.Linear(out_dim, 1),
                                 # nn.Sigmoid()  # When using sigmoid, change the criterion and loss on training.py
                                 )

        self.sigmoid = nn.Sigmoid()
        self.threshold_selector = nn.Parameter(torch.tensor(0.5))

    def forward(self, q, v):
        """
        Implementation based on CrossAttention from PoinTr
        Args:
            q:
            v:

        Returns:

        """
        k = v

        residual = q
        attention_output, _ = self.multihead_attention(q, k, v)
        attention_output += residual
        attention_output = self.layer_norm1(attention_output)

        attention_output = self.pointwise_feed(attention_output)
        attention_output += residual
        attention_output = self.layer_norm2(attention_output)

        mlp_in = torch.cat((q, attention_output), dim=2)
        out = self.mlp(mlp_in)
        out = out.squeeze(-1)

        occ = self.sigmoid(out)
        extracted_mask = (occ >= self.threshold_selector)
        return out, extracted_mask


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.partial_conv_onet = ConvONet(args['dualconvomendnet_args'])
        self.restoration_conv_onet = ConvONet(args['dualconvomendnet_args'])

        self.attention_net = AttentionMLP(args['dualconvomendnet_args']['c_dim'],
                                          args['num_points'],
                                          args['dualconvomendnet_args']['attention_kwargs']['num_heads'])
        # self.attention_net = GDP2(d_model=args['dualconvomendnet_args']['decoder_kwargs']['hidden_size'],
        #                           d_model_out=args['dualconvomendnet_args']['decoder_kwargs']['hidden_size'])

    @staticmethod
    def _filter_noise(noise, mask):
        filtered = torch.zeros(noise.shape, device=noise.device)
        sizes = mask.sum(axis=1)
        pad_sizes = 2048 - sizes
        for b in range(len(filtered)):
            filtered[b] = nn.functional.pad(noise[b, mask[b], :], (0, 0, 0, pad_sizes[b]), "constant", 42)
        return filtered

    def forward(self, p, partial, noise, inputs, gt_coarse=None, gt=None, is_training=True):
        partial = partial.transpose(1, 2).contiguous()
        noise = noise.transpose(1, 2).contiguous()

        partial_out = self.partial_conv_onet(partial, partial)
        restoration_out = self.restoration_conv_onet(noise, noise)

        logits, occ_mask = self.attention_net(partial_out, restoration_out)

        # override mask threshold
        # occ_mask = (torch.sigmoid(logits) >= 0.4)

        if is_training:
            loss3 = nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none')
            loss3 = loss3.mean(axis=1)

            filtered_list = self._filter_noise(noise, occ_mask)
            filtered_list_gt = self._filter_noise(noise, gt.bool())
            loss2, _ = calc_cd(filtered_list, filtered_list_gt)

            total_train_loss = loss3.mean()  # + loss2.mean()

            return loss3, loss2, total_train_loss
        else:
            filtered_list = self._filter_noise(noise, occ_mask)

            gt_is_occ = (len(gt.shape) == 2)  # it might be false, we may receive a point cloud ground truth
            if gt_is_occ:
                bce = nn.functional.binary_cross_entropy_with_logits(logits, gt, reduction='none').mean(axis=1)
                filtered_list_gt = self._filter_noise(noise, gt.bool())
                cd_p, cd_t = calc_cd(filtered_list, filtered_list_gt)
            else:
                bce = None
                cd_p, cd_t = calc_cd(filtered_list, gt)

            return {
                'out1': None,
                'out2': filtered_list,
                'occ': torch.sigmoid(logits),
                'cd_t': cd_t, 'cd_p': cd_p,
                'bce': bce,
            }


def train_step(data, net, do_summary_string):
    _, inputs, noise, gt, restoration_gt = data

    inputs = inputs.float().cuda()
    noise = noise.float().cuda()
    gt = gt.float().cuda()

    inputs = inputs.transpose(2, 1).contiguous()
    noise = noise.transpose(2, 1).contiguous()

    loss1, loss2, net_loss = net(None, inputs, noise, None, gt_coarse=restoration_gt, gt=gt)

    summary_string = None
    if do_summary_string:
        summary_string = (
            f' bce_loss: {loss1.mean().item()}'
            f' fine_loss: {loss2.mean().item()}'
            # f' coarse_loss: {loss3.mean().item()}'
            f' total_loss: {net_loss.mean().item()}'
        )

    return net_loss, summary_string


def val_step(data, net):
    _, inputs, noise, gt, restoration_gt = data

    inputs = inputs.float().cuda()
    noise = noise.float().cuda()
    gt = gt.float().cuda()

    inputs = inputs.transpose(2, 1).contiguous()
    noise = noise.transpose(2, 1).contiguous()

    result_dict = net(None, inputs, noise, None, gt_coarse=restoration_gt, gt=gt, is_training=False)
    return result_dict
