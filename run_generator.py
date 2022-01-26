# ------------------------------------------------------------------------
# Generator Script for 3D StyleGAN

# ------------------------------------------------------------------------
# Original StyleGAN2 Copyright
# ------------------------------------------------------------------------
# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import os 

import pretrained_networks

import nibabel as nib

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
#asd

#----------------------------------------------------------------------------

# def generate_images(network_pkl, seeds, truncation_psi):
#     print('Loading networks from "%s"...' % network_pkl)
#     _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
#     noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

#     Gs_kwargs = dnnlib.EasyDict()
#     Gs_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
#     Gs_kwargs.randomize_noise = False

#     if truncation_psi is not None:
#         Gs_kwargs.truncation_psi = truncation_psi

#     for seed_idx, seed in enumerate(seeds):
#         print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
#         rnd = np.random.RandomState(seed)
#         z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
#         tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
#         images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]

#         img = nib.Nifti1Image( images[0, :, :, :, 0 ], np.eye(4))

#         nib.save( img, dnnlib.make_run_dir_path('seed%04d.nii.gz' % seed) )

#         PIL.Image.fromarray(images[0, images.shape[1]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_x.png' % seed))
#         PIL.Image.fromarray(images[0, :, images.shape[2]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_y.png' % seed))
#         PIL.Image.fromarray(images[0, :, :, images.shape[3]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_z.png' % seed))


def generate_images(network_pkl, seeds, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
    Gs_syn_kwargs.randomize_noise = True
    Gs_syn_kwargs.minibatch_size = 1

    # Gs_kwargs = dnnlib.EasyDict()
    # Gs_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
    # Gs_kwargs.randomize_noise = False

    # if truncation_psi is not None:
    #     Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]

        w = Gs.components.mapping.run( z, None )
        w = w_avg + (w - w_avg) * truncation_psi # [minibatch, layer, component]
        images = Gs.components.synthesis.run( w, **Gs_syn_kwargs)

        # tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        # images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]

        img = nib.Nifti1Image( images[0, :, :, :, 0 ], np.eye(4))

        nib.save( img, dnnlib.make_run_dir_path('seed%04d.nii.gz' % seed) )

        PIL.Image.fromarray(images[0, images.shape[1]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_x.png' % seed))
        PIL.Image.fromarray(images[0, :, images.shape[2]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_y.png' % seed))
        PIL.Image.fromarray(images[0, :, :, images.shape[3]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('seed%04d_z.png' % seed))

#----------------------------------------------------------------------------

def style_mixing_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
    Gs_syn_kwargs.randomize_noise = True
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))

    # if os.path.exists( "/data/vision/polina/users/razvan/sungmin/stylegan2/_all_z_temp.npy" ):
    #     all_z = np.load( "/data/vision/polina/users/razvan/sungmin/stylegan2/_all_z_temp.npy" )
    # else:
    #     all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    #     np.save( "/data/vision/polina/users/razvan/sungmin/stylegan2/_all_z_temp.npy", all_z )

    # all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    # np.save( "/data/vision/polina/users/razvan/sungmin/stylegan2/_all_z_temp.npy", all_z )

    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    np.save( "_all_z_temp.npy", all_z )


    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]
    
    print( "=========================" )
    print( "all_z" )
    print( "=========================" )
    print( all_z )
    print( "=========================" )
    print( "all_w" )
    print( "=========================" )
    print( all_w )
    
    print( "=========================" )
    print( "all_z.shape" )
    print( "=========================" )
    print( all_z.shape )
    print( "=========================" )
    print( "all_w.shape" )
    print( "=========================" )
    print( all_w.shape )

    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating style-mixed images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            print( "=======================================" )
            print( image.shape )
            print( "========================================" )
            image_dict[(row_seed, col_seed)] = image

    print('Saving images...')
    for (row_seed, col_seed), image in image_dict.items():
        img = nib.Nifti1Image( image[ :, :, :, 0 ], np.eye(4))
        nib.save( img, dnnlib.make_run_dir_path('%d-%d.nii.gz' % (row_seed, col_seed)) )

        PIL.Image.fromarray(image[image.shape[0]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('%d-%d_x.png' % (row_seed, col_seed)))
        PIL.Image.fromarray(image[ :, image.shape[1]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('%d-%d_y.png' % (row_seed, col_seed)))
        PIL.Image.fromarray(image[ :, :, image.shape[2]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('%d-%d_z.png' % (row_seed, col_seed)))

        # PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    print('Saving image grid...')
    _N, _C, H, W, D = Gs.output_shape

    canvas_x = PIL.Image.new('L', ( W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas_x.paste(PIL.Image.fromarray(image_dict[key][ :, :, D//2, 0 ], 'L'), (W * col_idx, H * row_idx))
    canvas_x.save(dnnlib.make_run_dir_path('grid_x.png'))


    canvas_y = PIL.Image.new('L', (D * (len(col_seeds) + 1), W * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas_y.paste(PIL.Image.fromarray(image_dict[key][ H//2, :, :, 0 ], 'L'), (D * col_idx, W * row_idx))
    canvas_y.save(dnnlib.make_run_dir_path('grid_y.png'))

    canvas_z = PIL.Image.new('L', (D * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    for row_idx, row_seed in enumerate([None] + row_seeds):
        for col_idx, col_seed in enumerate([None] + col_seeds):
            if row_seed is None and col_seed is None:
                continue
            key = (row_seed, col_seed)
            if row_seed is None:
                key = (col_seed, col_seed)
            if col_seed is None:
                key = (row_seed, row_seed)
            canvas_z.paste(PIL.Image.fromarray(image_dict[key][ :, W//2, :, 0 ], 'L'), (D * col_idx, H * row_idx))
    canvas_z.save(dnnlib.make_run_dir_path('grid_z.png'))



#----------------------------------------------------------------------------

def interpolation_example(network_pkl, row_seeds, col_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')
    all_seeds = list(set(row_seeds + col_seeds))

    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]

    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]
    interp_ratio = np.linspace( 0.0, 1.0, num=10, endpoint=True )

    image_dict = {(seed, seed, interp_idx ): image for seed, interp_idx, image in zip(all_seeds, np.arange( len( interp_ratio ) ), list(all_images))}

    print('Generating interpolated images...')
    for row_seed in row_seeds:
        for col_seed in col_seeds:
            w_row = w_dict[ row_seed ].copy()
            w_col = w_dict[ col_seed ].copy()


            for interp_idx in range( len ( interp_ratio ) ):
                w = w_dict[row_seed].copy()
                w[ col_styles ] = ( 1.0 - interp_ratio[ interp_idx ] ) * w_row[ col_styles ] + ( interp_ratio[ interp_idx ] * w_col[ col_styles ] )

                image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
                image_dict[(row_seed, col_seed, interp_idx ) ] = image

    print('Saving images...')
    for (row_seed, col_seed, interp_idx ), image in image_dict.items():
        img = nib.Nifti1Image( image[ :, :, :, 0 ], np.eye(4))
        nib.save( img, dnnlib.make_run_dir_path('%d-%d-%d.nii.gz' % (row_seed, col_seed, interp_idx ) ) )

        PIL.Image.fromarray(image[image.shape[0]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('x-%d-%d-%d.png' % (row_seed, col_seed, interp_idx )))
        PIL.Image.fromarray(image[ :, image.shape[1]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('y-%d-%d-%d.png' % (row_seed, col_seed, interp_idx)))
        PIL.Image.fromarray(image[ :, :, image.shape[2]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('z-%d-%d-%d.png' % (row_seed, col_seed, interp_idx)))

        # PIL.Image.fromarray(image, 'RGB').save(dnnlib.make_run_dir_path('%d-%d.png' % (row_seed, col_seed)))

    # print('Saving image grid...')
    # _N, _C, H, W, D = Gs.output_shape

    # canvas_x = PIL.Image.new('L', ( W * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    # for row_idx, row_seed in enumerate([None] + row_seeds):
    #     for col_idx, col_seed in enumerate([None] + col_seeds):
    #         if row_seed is None and col_seed is None:
    #             continue
    #         key = (row_seed, col_seed)
    #         if row_seed is None:
    #             key = (col_seed, col_seed)
    #         if col_seed is None:
    #             key = (row_seed, row_seed)
    #         canvas_x.paste(PIL.Image.fromarray(image_dict[key][ :, :, D//2, 0 ], 'L'), (W * col_idx, H * row_idx))
    # canvas_x.save(dnnlib.make_run_dir_path('grid_x.png'))


    # canvas_y = PIL.Image.new('L', (D * (len(col_seeds) + 1), W * (len(row_seeds) + 1)), 'black')
    # for row_idx, row_seed in enumerate([None] + row_seeds):
    #     for col_idx, col_seed in enumerate([None] + col_seeds):
    #         if row_seed is None and col_seed is None:
    #             continue
    #         key = (row_seed, col_seed)
    #         if row_seed is None:
    #             key = (col_seed, col_seed)
    #         if col_seed is None:
    #             key = (row_seed, row_seed)
    #         canvas_y.paste(PIL.Image.fromarray(image_dict[key][ H//2, :, :, 0 ], 'L'), (D * col_idx, W * row_idx))
    # canvas_y.save(dnnlib.make_run_dir_path('grid_y.png'))

    # canvas_z = PIL.Image.new('L', (D * (len(col_seeds) + 1), H * (len(row_seeds) + 1)), 'black')
    # for row_idx, row_seed in enumerate([None] + row_seeds):
    #     for col_idx, col_seed in enumerate([None] + col_seeds):
    #         if row_seed is None and col_seed is None:
    #             continue
    #         key = (row_seed, col_seed)
    #         if row_seed is None:
    #             key = (col_seed, col_seed)
    #         if col_seed is None:
    #             key = (row_seed, row_seed)
    #         canvas_z.paste(PIL.Image.fromarray(image_dict[key][ :, W//2, :, 0 ], 'L'), (D * col_idx, H * row_idx))
    # canvas_z.save(dnnlib.make_run_dir_path('grid_z.png'))

#----------------------------------------------------------------------------

def average_image_example(network_pkl, num_seeds, truncation_psi, col_styles, minibatch_size=4):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    w_avg = Gs.get_var('dlatent_avg') # [component]

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_3d_images_to_uint8, nchwd_to_nhwdc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = minibatch_size

    print('Generating W vectors...')

    seeds = np.random.randint( 1, high=100000, size=num_seeds )

    all_seeds = list( seeds )

    all_z = np.stack([np.random.RandomState(seed).randn(*Gs.input_shape[1:]) for seed in all_seeds]) # [minibatch, component]
    all_w = Gs.components.mapping.run(all_z, None) # [minibatch, layer, component]

    all_w = w_avg + (all_w - w_avg) * truncation_psi # [minibatch, layer, component]
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))} # [layer, component]

    avg_w = np.average( all_w, axis=0 )

    print('Generating images...')
    all_images = Gs.components.synthesis.run(all_w, **Gs_syn_kwargs) # [minibatch, height, width, channel]

    image_dict = {(seed): image for seed, image in zip(all_seeds, list(all_images))}

    print('Generating interpolated images...')
    
    for seed in seeds:
        w = w_dict[ seed ].copy()

        image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
        image_dict[ ( seed ) ] = image

    print('Saving images...')
    for ( seed ), image in image_dict.items():
        img = nib.Nifti1Image( image[ :, :, :, 0 ], np.eye(4))
        nib.save( img, dnnlib.make_run_dir_path('%d.nii.gz' % ( seed ) ) )

        PIL.Image.fromarray(image[image.shape[0]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('x-%d.png' % ( seed ) ) )
        PIL.Image.fromarray(image[ :, image.shape[1]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('y-%d.png' % ( seed ) ) )
        PIL.Image.fromarray(image[ :, :, image.shape[2]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('z-%d.png' % ( seed ) ) )

    print( 'Generating the average image' )
    avg_gen_img = Gs.components.synthesis.run(avg_w[np.newaxis], **Gs_syn_kwargs)[0]

    avg_img = nib.Nifti1Image( avg_gen_img[ :, :, :, 0 ], np.eye(4))
    nib.save( avg_img, dnnlib.make_run_dir_path('average.nii.gz' ) )

    PIL.Image.fromarray(avg_gen_img[avg_gen_img.shape[0]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('x-average.png' ) )
    PIL.Image.fromarray(avg_gen_img[ :, avg_gen_img.shape[1]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('y-average.png' ) )
    PIL.Image.fromarray(avg_gen_img[ :, :, avg_gen_img.shape[2]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('z-average.png' ) )

    print( 'Linear interpolation from the average image to the generated images' )
    interp_ratio = np.linspace( 0.0, 1.0, num=10, endpoint=True )
    interp_image_dict = {( seed, interp_idx ): image for seed, interp_idx, image in zip(all_seeds, np.arange( len( interp_ratio ) ), list(all_images))}

    for seed in seeds:
        w = avg_w.copy()
        w_in = w_dict[ seed ].copy()

        for interp_idx in range( len( interp_ratio ) ):
            w[ col_styles ] = ( 1.0 - interp_ratio[ interp_idx ] ) * avg_w[ col_styles ] + ( interp_ratio[ interp_idx ] * w_in[ col_styles ] )

            image = Gs.components.synthesis.run(w[np.newaxis], **Gs_syn_kwargs)[0]
            interp_image_dict[ (seed, interp_idx ) ] = image

    for ( seed, interp_idx ), image in interp_image_dict.items():
        img = nib.Nifti1Image( image[ :, :, :, 0 ], np.eye(4))
        nib.save( img, dnnlib.make_run_dir_path('%d-%d.nii.gz' % (seed, interp_idx ) ) )

        PIL.Image.fromarray(image[image.shape[0]//2, :, :, 0 ], 'L').save(dnnlib.make_run_dir_path('x-%d-%d.png' % (seed, interp_idx ) ) )
        PIL.Image.fromarray(image[ :, image.shape[1]//2, :, 0 ], 'L').save(dnnlib.make_run_dir_path('y-%d-%d.png' % (seed, interp_idx ) ) )
        PIL.Image.fromarray(image[ :, :, image.shape[2]//2, 0 ], 'L').save(dnnlib.make_run_dir_path('z-%d-%d.png' % (seed, interp_idx ) ) )

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

  # Generate ffhq uncurated images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5

  # Generate ffhq curated images (matches paper Figure 11)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --seeds=66,230,389,1518 --truncation-psi=1.0

  # Generate uncurated car images (matches paper Figure 12)
  python %(prog)s generate-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=6000-6025 --truncation-psi=0.5

  # Generate style mixing example (matches style mixing video clip)
  python %(prog)s style-mixing-example --network=gdrive:networks/stylegan2-ffhq-config-f.pkl --row-seeds=85,100,75,458,1500 --col-seeds=55,821,1789,293 --truncation-psi=1.0
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('generate-images', help='Generate images')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_style_mixing_example = subparsers.add_parser('style-mixing-example', help='Generate style mixing video')
    parser_style_mixing_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_style_mixing_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_style_mixing_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_style_mixing_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_style_mixing_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_style_mixing_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_interpolation_example = subparsers.add_parser('interpolation-example', help='Generate interpolation video')
    parser_interpolation_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_interpolation_example.add_argument('--row-seeds', type=_parse_num_range, help='Random seeds to use for image rows', required=True)
    parser_interpolation_example.add_argument('--col-seeds', type=_parse_num_range, help='Random seeds to use for image columns', required=True)
    parser_interpolation_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_interpolation_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_interpolation_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_interpolation_example = subparsers.add_parser('average-image-example', help='Generate interpolation video')
    parser_interpolation_example.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_interpolation_example.add_argument('--num-seeds', type=int, help='Number of random seeds to use for image generation (Equal to the number of images)', required=True)
    parser_interpolation_example.add_argument('--col-styles', type=_parse_num_range, help='Style layer range (default: %(default)s)', default='0-6')
    parser_interpolation_example.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_interpolation_example.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')


    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'generate-images': 'run_generator.generate_images',
        'style-mixing-example': 'run_generator.style_mixing_example',
        'interpolation-example': 'run_generator.interpolation_example',
        'average-image-example': 'run_generator.average_image_example'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
