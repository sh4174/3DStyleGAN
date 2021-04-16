# Please see the license information of the original StyleGAN2 in https://nvlabs.github.io/stylegan2/license.html
import argparse
import copy
import os
import sys

import dnnlib
from dnnlib import EasyDict

from metrics.metric_defaults import metric_defaults

#----------------------------------------------------------------------------
_valid_configs = [
    'Gorig-Dres-3d-1mm-base567',
    'Gorig-Dorig-3d-1mm-base567',
    'Gorig-Dorig-r1-3d-1mm-base567',
    'Gorig-Dres-3d-2mm-base567',
    'Gorig-Dres-R1-3d-2mm-base567',
    'Gorig-Dres-DeepFil-R1-3d-2mm-base567',
    'Gorig-Dres-R2-3d-2mm-base567',
]

#----------------------------------------------------------------------------

def run(dataset, data_dir, result_dir, config_id, num_gpus, total_kimg, gamma, mirror_augment, metrics, resume_pkl):
    train     = EasyDict(run_func_name='training.training_loop.training_loop') # Options for training loop.
    G         = EasyDict(func_name='training.networks_stylegan2.G_main')       # Options for generator network.
    D         = EasyDict(func_name='training.networks_stylegan2.D_stylegan2')  # Options for discriminator network.
    G_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for generator optimizer.
    D_opt     = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8)                  # Options for discriminator optimizer.
    G_loss    = EasyDict(func_name='training.loss.G_logistic')      # Options for generator loss.
    D_loss    = EasyDict(func_name='training.loss.D_logistic_r1')              # Options for discriminator loss.
    sched     = EasyDict()                                                     # Options for TrainingSchedule.
    grid      = EasyDict(size='1080p', layout='random')                           # Options for setup_snapshot_image_grid().
    sc        = dnnlib.SubmitConfig()                                          # Options for dnnlib.submit_run().
    tf_config = {'rnd.np_random_seed': 100}                                   # Options for tflib.init_tf().

    if config_id == 'Gorig-Dres-3d-1mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r1')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 18
        G.fmap_max = 18
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 18
        D.fmap_max = 18
        D.base_size = [ 5, 6, 7 ]

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1

        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 2
        sched.minibatch_gpu_dict = {8: 8, 16: 8, 32: 2, 64: 2, 128:2}
        # sched.minibatch_gpu_dict = {8: 8, 16: 8, 32: 4, 64: 4}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus, 128: sched.minibatch_gpu_dict[ 128 ] * num_gpus}
        # sched.minibatch_size_dict = {8: 32, 16: 32, 32: 16, 64: 16}
    elif config_id == 'Gorig-Dorig-3d-1mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 16
        G.fmap_max = 16
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'orig'        
        # D.resolution=128
        D.fmap_min = 16
        D.fmap_max = 16
        D.base_size = [ 5, 6, 7 ]

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1

        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 2
        sched.minibatch_gpu_dict = {8: 8, 16: 4, 32: 4, 64: 2, 128:2}
        # sched.minibatch_gpu_dict = {8: 8, 16: 8, 32: 4, 64: 4}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus, 128: sched.minibatch_gpu_dict[ 128 ] * num_gpus}
        # sched.minibatch_size_dict = {8: 32, 16: 32, 32: 16, 64: 16}
    elif config_id == 'Gorig-Dorig-r1-3d-1mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r1')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 24
        G.fmap_max = 24
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'orig'        
        # D.resolution=128
        D.fmap_min = 24
        D.fmap_max = 24
        D.base_size = [ 5, 6, 7 ]

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1

        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 2
        sched.minibatch_gpu_dict = {8: 8, 16: 4, 32: 4, 64: 2, 128:2}
        # sched.minibatch_gpu_dict = {8: 8, 16: 8, 32: 4, 64: 4}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus, 128: sched.minibatch_gpu_dict[ 128 ] * num_gpus}
        # sched.minibatch_size_dict = {8: 32, 16: 32, 32: 16, 64: 16}
    elif config_id == 'Gorig-Dres-3d-2mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 96
        G.fmap_max = 96
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 96
        D.fmap_max = 96
        D.base_size = [ 5, 6, 7 ]
    

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1

        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 8
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 16, 64: 8}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus}
    elif config_id == 'Gorig-Dres-3d-2mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r1')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 96
        G.fmap_max = 96
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 96
        D.fmap_max = 96
        D.base_size = [ 5, 6, 7 ]
        
        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1

        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 8
        sched.minibatch_gpu_dict = {8: 32, 16: 16, 32: 16, 64: 8}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus}
    elif config_id == 'Gorig-Dres-R1-3d-2mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r1')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 32
        G.fmap_max = 32
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 32
        D.fmap_max = 32
        D.base_size = [ 5, 6, 7 ]
        

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1
        train.lazy_regularization = False 
        
        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 2
        sched.minibatch_gpu_dict = {8: 16, 16: 8, 32: 4, 64: 2}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus}
    elif config_id == 'Gorig-Dres-R2-3d-2mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r2')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 96
        G.dlatent_size = 96
        G.mapping_fmaps = 96

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 16
        G.fmap_max = 16
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 16
        D.fmap_max = 16
        D.base_size = [ 5, 6, 7 ]
    
        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1
        train.lazy_regularization = False 
        
        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 16
        sched.minibatch_gpu_dict = {8: 16, 16: 8, 32: 4, 64: 2}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus}
    elif config_id == 'Gorig-Dres-DeepFil-R1-3d-2mm-base567':
        train   = EasyDict(run_func_name='training.training_loop_3d.training_loop') # Options for training loop.

        G       = EasyDict(func_name='training.networks3d_stylegan2.G_main')
        D       = EasyDict(func_name='training.networks3d_stylegan2.D_stylegan2_3d_curated_real')

        G_loss  = EasyDict(func_name='training.loss.G_logistic_ns_pathreg')
        D_loss  = EasyDict(func_name='training.loss.D_logistic_r1')

        # Generator Params 
        G.architecture = 'orig'
        
        # Mapping Network Params
        G.latent_size = 128
        G.dlatent_size = 128
        G.mapping_fmaps = 128

        # Synthesis Network Params
        # G.resolution = 128
        G.fmap_min = 128
        G.fmap_max = 128
        G.base_size = [ 5, 6, 7 ]

        # Discriminator Params 
        D.architecture = 'resnet'        
        # D.resolution=128
        D.fmap_min = 128
        D.fmap_max = 128
        D.base_size = [ 5, 6, 7 ]
        

        train.data_dir = data_dir
        train.total_kimg = total_kimg
        train.mirror_augment = mirror_augment
        train.image_snapshot_ticks = 1
        train.network_snapshot_ticks = 1
        train.lazy_regularization = False 
        
        sched.G_lrate_base = sched.D_lrate_base = 0.002

        sched.minibatch_gpu_base = 8
        sched.minibatch_gpu_dict = {8: 8, 16: 8, 32: 4, 64: 2}

        sched.minibatch_size_base = sched.minibatch_gpu_base * num_gpus
        sched.minibatch_size_dict = {8: sched.minibatch_gpu_dict[ 8 ] * num_gpus , 16:  sched.minibatch_gpu_dict[ 16 ] * num_gpus, 32:  sched.minibatch_gpu_dict[ 32 ] * num_gpus, 64:  sched.minibatch_gpu_dict[ 64 ] * num_gpus}
    else:
        print( "Unknown Config" )
        return

    # D_loss.gamma = 10
    metrics = [metric_defaults[x] for x in metrics]
    desc = '3dstylegan2'

    desc += '-' + dataset
    dataset_args = EasyDict(tfrecord_dir=dataset)

    assert num_gpus in [1, 2, 4, 8]
    sc.num_gpus = num_gpus
    desc += '-%dgpu' % num_gpus

    assert config_id in _valid_configs
    desc += '-' + config_id

    G.fmap_base = D.fmap_base = 8 << 10
    
    if gamma is not None:
        D_loss.gamma = gamma

    # print( "======================================" )
    # print( "======         Dataset Arg      ======" )
    # print( "======================================" )
    # print( dataset_args )

    # print( "======================================" )
    # print( "======         Desc             ======" )
    # print( "======================================" )
    # print( desc )
    # print( "======================================" )

    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    kwargs = EasyDict(train)
    kwargs.update(G_args=G, D_args=D, G_opt_args=G_opt, D_opt_args=D_opt, G_loss_args=G_loss, D_loss_args=D_loss)
    kwargs.update(dataset_args=dataset_args, sched_args=sched, grid_args=grid, metric_arg_list=metrics, tf_config=tf_config, resume_pkl=resume_pkl)
    kwargs.submit_config = copy.deepcopy(sc)
    kwargs.submit_config.run_dir_root = result_dir
    kwargs.submit_config.run_desc = desc

    dnnlib.submit_run(**kwargs)

#----------------------------------------------------------------------------

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def _parse_comma_sep(s):
    if s is None or s.lower() == 'none' or s == '':
        return []
    return s.split(',')

#----------------------------------------------------------------------------

_examples = '''examples:

  # Train StyleGAN2 using the FFHQ dataset
  python %(prog)s --num-gpus=8 --data-dir=~/datasets --config=config-f --dataset=ffhq --mirror-augment=true

valid configs:

  ''' + ', '.join(_valid_configs) + '''

valid metrics:

  ''' + ', '.join(sorted([x for x in metric_defaults.keys()])) + '''

'''

def main():
    parser = argparse.ArgumentParser(
        description='Train StyleGAN2.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    parser.add_argument('--dataset', help='Training dataset', required=True)
    parser.add_argument('--config', help='Training config (default: %(default)s)', default='config-f', required=True, dest='config_id', metavar='CONFIG')
    parser.add_argument('--num-gpus', help='Number of GPUs (default: %(default)s)', default=1, type=int, metavar='N')
    parser.add_argument('--total-kimg', help='Training length in thousands of images (default: %(default)s)', metavar='KIMG', default=25000, type=int)
    parser.add_argument('--gamma', help='R1 regularization weight (default is config dependent)', default=None, type=float)
    parser.add_argument('--mirror-augment', help='Mirror augment (default: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)
    parser.add_argument('--metrics', help='Comma-separated list of metrics or "none" (default: %(default)s)', default='mmd_test', type=_parse_comma_sep)
    parser.add_argument('--resume-pkl', help='Initialise training from a pre-trained network, as .pkl (default: %(default)s)', default=None)

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print ('Error: dataset root directory does not exist.')
        sys.exit(1)

    if args.config_id not in _valid_configs:
        print ('Error: --config value must be one of: ', ', '.join(_valid_configs))
        sys.exit(1)

    for metric in args.metrics:
        if metric not in metric_defaults:
            print ('Error: unknown metric \'%s\'' % metric)
            sys.exit(1)

    import socket
    os.system("printf '\033]2;%s-%s-%dgpus\033\\'" % (socket.gethostname(), args.dataset, args.num_gpus))

    run(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

