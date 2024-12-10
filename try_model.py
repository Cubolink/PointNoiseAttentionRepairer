"""
Uses the net to mend, restore, a point cloud.
"""

import logging
import os
import sys
import importlib
import argparse
import munch
import yaml
from utils.train_utils import *
from utils.test_utils import *
from dataset import (
    SimpleDataset,
    SimpleDatasetWithNoise
)
from matplotlib.pyplot import get_cmap
import trimesh


def repair(datapath):
    if args.model_name == 'PointAttN':
        dataset_test = SimpleDataset(datapath, file_extensions='.obj')
    else:
        dataset_test = SimpleDatasetWithNoise(datapath, file_extensions='.obj')

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    logging.info('Repairing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if args.model_name == 'PointAttN':
                inputs_cpu, obj = data
                inputs = inputs_cpu.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()
                result_dict = net(inputs, is_training=False)
            else:
                inputs_cpu, noise_cpu, obj = data

                inputs = inputs_cpu.float().cuda()
                noise = noise_cpu.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()
                noise = noise.transpose(2, 1).contiguous()
                if args.model_name == 'DualConvOMendNet':
                    raise NotImplementedError
                else:
                    result_dict = net(inputs, noise, is_training=False)

            for j in range(len(obj)):
                label = os.path.relpath(obj[j], datapath)  # subfolder/example.off
                label = os.path.splitext(label)[0]  # subfolder/example
                path = os.path.join(os.path.dirname(args.load_model), 'repair', str(label))
                if not os.path.isdir(path):
                    os.makedirs(path)
                path = os.path.join(path, str(os.path.splitext(os.path.basename(obj[j]))[0]) + '.obj')

                save_obj(inputs[j].transpose(0, 1), path.replace('.obj', '_inputs.obj'))
                if args.model_name == 'PointAttNB' or args.model_name == 'DualConvOMendNet':
                    # the output comes already filtered
                    mask = (result_dict['out2'][j] < 1).all(axis=1)
                    save_obj(
                        result_dict['out2'][j][mask],
                        path.replace('.obj', '_out.obj')
                    )
                    save_obj(
                        torch.cat([inputs[j].transpose(0, 1), result_dict['out2'][j][mask]], dim=0),
                        path.replace('.obj', '_out+inputs.obj')
                    )
                    # color noise using predicted occupancy values
                    cmap = get_cmap('gray')
                    trimesh.PointCloud(
                        noise[j].transpose(0, 1).cpu().numpy(),
                        colors=cmap(result_dict['occ'][j].cpu().numpy())
                    ).export(
                        path.replace('.obj', '_occ.obj')
                    )
                else:
                    save_obj(
                        result_dict['out2'][j],
                        path.replace('.obj', '_out.obj')
                    )
                    save_obj(
                        torch.cat([inputs[j].transpose(0, 1), result_dict['out2'][j]]),
                        path.replace('.obj', '_out+inputs.obj')
                    )
                if result_dict['out1'] is not None:
                    save_obj(result_dict['out1'][j], path.replace('.obj', '_coarse.obj'))
                    save_obj(
                        torch.cat([inputs[j].transpose(0, 1), result_dict['out1'][j]], dim=0),
                        path.replace('.obj', '_coarse+inputs.obj'))

                # save_obj(gt[j], path.replace('.obj', '_gt.obj'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-d', '--datapath', help='path to the data to repair', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs', arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)

    repair(arg.datapath)
