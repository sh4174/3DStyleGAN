# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import projector
import pretrained_networks
from training import dataset
from training import misc

from forwardModels import *

def constructForwardModel(recontype, imgSize):
  if recontype == 'none':
    forward = ForwardNone(); forwardTrue = forward # no forward model, just image inversion

  elif recontype == 'super-resolution':
    # Create downsampling forward corruption model
    forward = ForwardDownsample(res = 32); forwardTrue = forward # res = target resolution

  elif recontype == 'inpaint':
    # Create forward model that fills-in part of image with zeros (change the height/width to control the bounding box)
    forward = ForwardFillMask();  forwardTrue = ForwardFillBoundingBox(offset_height=int(0.3 * imgSize), offset_width=int(0.3 * imgSize), target_height=int(0.3 * imgSize), target_width=int(0.3 * imgSize), opt_params = False) # res = target resolution

  elif recontype == 'k-space-cs': # k-space compressed sensing task
    # create forward model that applyes FFT, drops random lines in k-space, then applies inverse FFT to obtain corrupted image
    forward = ForwardUndersampleMRI(img_size=imgSize, corruption_frac = 0.8); forwardTrue = forward
  else:
    raise ValueError('recontype has to be either none, super-resolution, inpaint, or k-space-cs')
  
  return forward, forwardTrue

def getImgSize(Gs):
  Gs_kwargs = dnnlib.EasyDict()
  Gs_kwargs.randomize_noise = False
  Gs_kwargs.truncation_psi = 0.5
  #print(Gs.components.synthesis.resolution)

  noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
  rnd = np.random.RandomState(0)
  z = rnd.randn(1, *Gs.input_shape[1:])
  tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
  images = Gs.run(z, None, **Gs_kwargs)
  print(images.shape)
  #asd

  return images.shape[2]
     
  
#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    print('target.shape', targets.shape)
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'corrupted-step%04d.png' % proj.get_cur_step(), drange=[-1,1])
            misc.save_image_grid(proj.get_clean_images(), png_prefix + 'clean-step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def recon_generated_images(network_pkl, seeds, num_snapshots, truncation_psi, recontype):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    
    imgSize = getImgSize(Gs)
    forward, forwardTrue = constructForwardModel(recontype, imgSize)

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        # create seed specific projector, in case forward model is different for each image
  
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        png_prefix = dnnlib.make_run_dir_path('seed%04d-' % seed)
        misc.save_image_grid(images, png_prefix + 'true.png', drange=[-1,1])
        print('images', type(images))
        #proj.forward.initVars()
        imagesCorrupted = forwardTrue(tf.convert_to_tensor(images)).eval()
        #images_k, images_k_corrupted = forwardTrue.get_k(tf.convert_to_tensor(images))
        #misc.save_image_grid(tf.stack([images_k, images_k_corrupted]).eval(), png_prefix + 'kspace.png', drange=[-1,1])
        #print('images_k', images_k[:,0])
        #print('images_k_corupted', images_k_corrupted[:,0])
        #asd
        #print('imagesCorrupted', type(imagesCorrupted))
        #print('images', images[:,0])
        #print('imagesCorrupt', imagesCorrupted.shape)
        #asd
        forward.calcMaskFromImg(imagesCorrupted)
        #print('forward.mask', forward.mask)
        proj = projector.Projector(forward)
        proj.set_network(Gs)
        project_image(proj, targets=imagesCorrupted, png_prefix=png_prefix, num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def recon_real_images(network_pkl, dataset_name, data_dir, num_images, num_snapshots, recontype):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    imgSize = getImgSize(Gs)
    forward, forwardTrue = constructForwardModel(recontype, imgSize)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])

        # save true image
        png_prefix = dnnlib.make_run_dir_path('image%04d-' % image_idx)
        misc.save_image_grid(images, png_prefix + 'true.png', drange=[-1,1])

        # generate corrupted image, with true forward corruption model
        imagesCorrupted = forwardTrue(tf.convert_to_tensor(images)).eval()
        forward.calcMaskFromImg(imagesCorrupted)
        proj = projector.Projector(forward)
        proj.set_network(Gs)
        project_image(proj, targets=imagesCorrupted, png_prefix=png_prefix, num_snapshots=num_snapshots)


#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Recon generated images
  python %(prog)s recon-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Recon real images
  python %(prog)s recon-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    recon_generated_images_parser = subparsers.add_parser('recon-generated-images', help='Project generated images')
    recon_generated_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    recon_generated_images_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    recon_generated_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    recon_generated_images_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    recon_generated_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    recon_generated_images_parser.add_argument('--recontype', help='Type of reconstruction: "none" (normal image inversion), "super-resolution", "inpaint", "k-space-cs" (default: %(default)s)', default='none', metavar='DIR')

    recon_real_images_parser = subparsers.add_parser('recon-real-images', help='Project real images')
    recon_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    recon_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    recon_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', default='datasets')
    recon_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=2)
    recon_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    recon_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    recon_real_images_parser.add_argument('--recontype', help='Type of reconstruction: "none" (normal image inversion), "super-resolution", "in-painting", "k-space-cs" (default: %(default)s)', default='none', metavar='DIR')

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')

    func_name_map = {
        'recon-generated-images': 'recon.recon_generated_images',
        'recon-real-images': 'recon.recon_real_images'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
