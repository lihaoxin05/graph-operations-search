import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--root_path',
        default='',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='',
        type=str,
        help='Path of video frames')
    parser.add_argument(
        '--proposal_path',
        default='',
        type=str,
        help='Path of proposal file')
    parser.add_argument(
        '--annotation_path',
        default='',
        type=str,
        help='Path of annotation file')
    parser.add_argument(
        '--result_path',
        default='',
        type=str,
        help='Path of results')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Path of saved checkpoint (.pth) to resume training')
    parser.add_argument(
        '--pretrain_path', 
        default='', 
        type=str, 
        help='Path of pretrained backbone (.pth) to initiablize')
    parser.add_argument(
        '--dataset',
        default='SomethingSomethingV1',
        type=str,
        help='Dataset (SomethingSomethingV1 | SomethingSomethingV2)')
    parser.add_argument(
        '--n_classes',
        default=174,
        type=int,
        help='Number of classes')
    parser.add_argument(
        '--sample_size',
        default=224,
        type=int,
        help='Height and width of input frames')
    parser.add_argument(
        '--sample_duration',
        default=32,
        type=int,
        help='Temporal duration of input clips')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale frame cropping')
    parser.add_argument(
        '--n_scales',
        default=2,
        type=int,
        help='Number of scales for multiscale frame cropping')
    parser.add_argument(
        '--scale_step',
        default=0.9,
        type=float,
        help='Scale step for multiscale frame cropping')
    parser.add_argument(
        '--train_crop',
        default='random',
        type=str,
        help='Spatial cropping method in training. Random is uniform. Corner is selected from 4 corners and 1 center. (random | corner | center)')
    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float,
        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--lr_patience',
        default=5,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )
    parser.add_argument('--momentum', 
        default=0.9, 
        type=float, 
        help='Momentum')
    parser.add_argument('--nesterov', 
        action='store_true', 
        help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument('--dampening', 
        default=0, 
        type=float, 
        help='dampening of SGD')
    parser.add_argument(
        '--weight_decay', 
        default=1e-3, 
        type=float, 
        help='Weight Decay')
    parser.add_argument('--batch_size', 
        default=64, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=50,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--n_val_samples',
        default=2,
        type=int,
        help='Number of validation samples for each video')
    parser.add_argument(
        '--n_test_samples',
        default=5,
        type=int,
        help='Number of test samples for each video')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--save_test_result',
        action='store_true',
        help='If true, save the test scores.')
    parser.set_defaults(save_test_result=False)
    parser.add_argument(
        '--no_cuda', 
        action='store_true', 
        help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=8,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=5,
        type=int,
        help='Trained model is saved at every this epochs.')
    parser.add_argument(
        '--log_step',
        default=20,
        type=int,
        help='Log is printed at every this step.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (50 | 101 | 152)')
    parser.add_argument(
        '--basenet_fixed_layers',
        default=4,
        type=int,
        help='Fix number of layers of backbone.')
    parser.add_argument(
        '--n_box_per_frame',
        default=10,
        type=int,
        help='Number of proposal per frame.')
    parser.add_argument(
        '--step_per_layer',
        default=3,
        type=int,
        help='Step per cell for NAS.')
    parser.add_argument(
        '--arch_learning_rate',
        default=1e-4,
        type=float,
        help='Architecture learning rate')
    parser.add_argument(
        '--arch_weight_decay', 
        default=1e-3, 
        type=float, 
        help='Weight Decay for architecture parameters')
    parser.add_argument(
        '--op_loss_weight', 
        default=1e-3, 
        type=float, 
        help='Loss weight for architecture')
    parser.add_argument(
        '--manual_seed', 
        default=1, 
        type=int, 
        help='Manually set random seed')
    args = parser.parse_args()

    return args
