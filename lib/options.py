import argparse

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # Datasets related
        g_data = parser.add_argument_group('Data')
        g_data.add_argument('--data', type=str, default='./data',
                            help='path to images (data folder)')
        # Training related
        g_train = parser.add_argument_group('Training')
        g_train.add_argument('--gpu_id', type=int, default=0, help='gpu id for cuda')
        g_train.add_argument('--num_threads', default=8, type=int, help='# sthreads for loading data')

        g_train.add_argument('--batch_size', type=int, default=12, help='input batch size')
        g_train.add_argument('--learning_rate', type=float, default=1e-4, help='adam learning rate')
        g_train.add_argument('--num_epoch', type=int, default=1001, help='num epoch to train')

        # Testing related
        g_test = parser.add_argument_group('Testing')
        g_test.add_argument('--img_mode', type=int, default=0, help='mode of image size')
        g_test.add_argument('--test_data', type=str, default='', help='test data path')
        
        # Model related
        g_model = parser.add_argument_group('Model')
        # General
        g_model.add_argument('--norm', type=str, default='group',
                             help='instance normalization or batch normalization or group normalization')
        # hg filter specify
        g_model.add_argument('--num_stack', type=int, default=4, help='# of hourglass')
        g_model.add_argument('--num_hourglass', type=int, default=2, help='# of stacked layer of hourglass')
        g_model.add_argument('--skip_hourglass', action='store_true', help='skip connection in hourglass')
        g_model.add_argument('--hg_down', type=str, default='ave_pool', help='ave pool || conv64 || conv128')
        g_model.add_argument('--hourglass_dim', type=int, default='256', help='256 | 512')

        # Classification General
        g_model.add_argument('--mlp_dim', nargs='+', default=[256, 1024, 512, 256, 128, 1], type=int,
                             help='# of dimensions of mlp')
        g_model.add_argument('--mlp_dim_orien', nargs='+', default=[256, 1024, 512, 256, 128, 3],
                             type=int, help='# of dimensions of orien mlp')
        g_model.add_argument('--mlp_dim_length', nargs='+', default=[256, 1024, 512, 256, 128, 1],
                             type=int, help='# of dimensions of length mlp, last layer= 8 when multi-class')
        g_model.add_argument('--len_class_num', type=int, default=8, help='# of strand length class')
        g_model.add_argument('--use_tanh', action='store_true',
                             help='using tanh after last conv of image_filter network')
        g_model.add_argument('--len_class_thres', type=float, default=0.5, help='threshold of length classifier')

        # for train
        parser.add_argument('--gen_orien', action='store_true', help='if generate orientation')
        parser.add_argument('--use_orien2d', action='store_true', help='if use 2d orientation map')
        parser.add_argument('--not_use_PEDE', action='store_true', help='if not use positional encoding and direction encoding')
        parser.add_argument('--not_use_DE', action='store_true', help='if not use direction encoding')
        parser.add_argument('--no_residual', action='store_true', help='no skip connection in mlp')
        parser.add_argument('--schedule', type=int, nargs='+', default=[200],
                            help='Decrease learning rate at these epochs.')
        parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

        # for eval
        parser.add_argument('--val_test_error', action='store_true', help='validate errors of test data')
        parser.add_argument('--val_train_error', action='store_true', help='validate errors of train data')
        parser.add_argument('--test_fix_len', action='store_true', help='if use fix length to test')

        # path
        parser.add_argument('--ckpt_path', type=str, default='./checkpoints', help='path to save checkpoints')
        parser.add_argument('--results_path', type=str, default='./results', help='path to save results ply')

        # special tasks
        self.initialized = True
        
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        opt = self.gather_options()
        return opt
