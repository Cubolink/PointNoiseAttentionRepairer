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
    GeometricBreaksDatasetNoNoise,
    GeometricBreaksDatasetWithNoise,
    GeometricBreaksDatasetWithNoiseOccupancy,
    GeometricBreaksDatasetWithMixedNoiseOccupancy
)
from matplotlib.pyplot import get_cmap
import trimesh

def test():
    if args.model_name == 'PointAttNA':
        dataset_test = GeometricBreaksDatasetWithNoise(args.chspath, prefix="test")
    elif args.model_name == 'PointAttNB':
        dataset_test = GeometricBreaksDatasetWithNoiseOccupancy(args.chspath, prefix="test")
    elif args.model_name == 'DualConvOMendNet':
        dataset_test = GeometricBreaksDatasetWithMixedNoiseOccupancy(args.chspath, prefix="test")
    else:
        dataset_test = GeometricBreaksDatasetNoNoise(args.chspath, prefix="test")

    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                                  shuffle=False, num_workers=int(args.workers))
    dataset_length = len(dataset_test)
    logging.info('Length of test dataset:%d', len(dataset_test))

    # load model
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    net.module.load_state_dict(torch.load(args.load_model)['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)
    net.eval()

    metrics = ['cd_t', 'cd_p', 'cd_t_coarse', 'cd_p_coarse']
    if args.model_name == 'DualConvOMendNet':
        metrics = ['cd_t', 'cd_p']
    test_loss_meters = {m: AverageValueMeter() for m in metrics}
    test_loss_cat = torch.zeros([len(dataset_test.label_map), len(metrics)], dtype=torch.float32).cuda()
    cat_num = torch.ones([len(dataset_test.label_map), 1], dtype=torch.float32).cuda() * 150

    logging.info('Testing...')

    with torch.no_grad():
        for i, data in enumerate(dataloader_test):
            if args.model_name == 'PointAttN':
                label, inputs_cpu, complete_gt_cpu, obj = data
                inputs = inputs_cpu.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()
                result_dict = net(inputs, gt=complete_gt_cpu, is_training=False)
            else:
                label, inputs_cpu, noise_cpu, complete_gt_cpu, restoration_gt_cpu, obj = data

                inputs = inputs_cpu.float().cuda()
                noise = noise_cpu.float().cuda()
                inputs = inputs.transpose(2, 1).contiguous()
                noise = noise.transpose(2, 1).contiguous()
                if args.model_name == 'DualConvOMendNet':
                    result_dict = net(None, inputs, noise, None, gt_coarse=restoration_gt_cpu, gt=complete_gt_cpu, is_training=False)
                else:
                    result_dict = net(inputs, noise, gt_coarse=restoration_gt_cpu, gt=complete_gt_cpu, is_training=False)
            for k, v in test_loss_meters.items():
                v.update(result_dict[k].mean().item())

            for j, l in enumerate(label):
                for ind, m in enumerate(metrics):
                    test_loss_cat[dataset_test.label_map[l], ind] += result_dict[m][int(j)]

            if i % args.step_interval_to_print == 0:
                logging.info('test [%d/%d]' % (i, dataset_length / args.batch_size))

            if args.save_vis:
                for j in range(len(label)):
                    path = os.path.join(os.path.dirname(args.load_model), args.dataset, 'all', str(label[j]))
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    path = os.path.join(path, str(obj[j]) + '.obj')

                    save_obj(inputs[j].transpose(0, 1), path.replace('.obj', '_inputs.obj'))
                    save_obj(complete_gt_cpu[j], path.replace('.obj', '_complete-gt.obj'))
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

        category_log = 'Loss per category:\n'
        for i in range(len(dataset_test.label_map)):
            category_log += '\ncategory name: %s' % (dataset_test.label_map_inverse[i])
            for ind, m in enumerate(metrics):
                scale_factor = 1 if m == 'f1' else 10000
                category_log += ' %s: %f' % (m, test_loss_cat[i, ind] / cat_num[i] * scale_factor)
        print(category_log)
        logging.info(category_log)

        overview_log = 'Overview results:\n'
        for metric, meter in test_loss_meters.items():
            overview_log += '%s: %f ' % (metric, meter.avg)
        logging.info(overview_log)
        print(overview_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = os.path.join('./cfgs', arg.config)
    args = munch.munchify(yaml.safe_load(open(config_path)))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if not args.load_model:
        raise ValueError('Model path must be provided to load model!')

    exp_name = os.path.basename(args.load_model)
    log_dir = os.path.join(os.path.dirname(args.load_model), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'test.log')),
                                                      logging.StreamHandler(sys.stdout)])

    test()
