# Multi-Scale Structural Similarity


import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

import scipy.ndimage as ndimage

 

def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    # img1, img2 : numpy arrays   
    mu1 = ndimage.gaussian_filter(img1, sd)
    mu2 = ndimage.gaussian_filter(img2, sd)
    
    mu1_sq = np.multiply( mu1, mu1 )
    mu2_sq = np.multiply( mu2, mu2 )
    mu1_mu2 = np.multiply( mu1, mu2 )

    sigma1_sq = ndimage.gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = ndimage.gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = ndimage.gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = np.mean(v1 / v2)  # contrast sensitivity

#     ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    return np.mean(ssim_map),cs



def msssim_3d(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    weights = tf.convert_to_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333] )

    img1 = tf.convert_to_tensor( img1 )
    img2 = tf.convert_to_tensor( img2 )

    img1 = tf.transpose(img1, [0, 2, 3, 4, 1])
    img2 = tf.transpose(img2, [0, 2, 3, 4, 1])

    levels = weights.shape[0]
    mssim = []
    mcs = []

    for l in range(levels):
        sim, cs = ssim_exact( tflib.tensor_to_numpy( img1 ), tflib.tensor_to_numpy( img2 ) )
        mssim.append(sim)
        mcs.append(cs)

        img1 = tf.nn.avg_pool3d( img1, ksize=2, strides=2, padding="VALID", data_format="NDHWC" )
        img2 = tf.nn.avg_pool3d( img2, ksize=2, strides=2, padding="VALID", data_format="NDHWC" )

    mssim = np.asarray(mssim)
    mcs = np.asarray(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = tflib.tensor_to_numpy( tf.math.reduce_prod(pow1[:-1] * pow2[-1]) )
    return output

# def ssim_lcs(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
#     # img1, img2 : numpy arrays   
#     C3 = C2 / 2

#     mu1 = ndimage.gaussian_filter(img1, sd)
#     mu2 = ndimage.gaussian_filter(img2, sd)
    
#     mu1_sq = np.multiply( mu1, mu1 )
#     mu2_sq = np.multiply( mu2, mu2 )
#     mu1_mu2 = np.multiply( mu1, mu2 )

#     sigma1 = np.sqrt( ndimage.gaussian_filter(img1 * img1, sd) - mu1_sq )
#     sigma2 = np.sqrt( ndimage.gaussian_filter(img2 * img2, sd) - mu2_sq )
#     sigma12 = ndimage.gaussian_filter(img1 * img2, sd) - mu1_mu2

#     l_map = ( 2 * mu1 * mu2 + C1 ) / ( mu1_sq + mu2_sq + C1 )

#     c_map = ( 2 * sigma1 * sigma2 + C2 ) / ( sigma1 * sigma1 + sigma2 * sigma2 + C2 )

#     s_map = ( sigma12 + C3 ) / ( sigma1 * sigma2 + C3 )

#     return l_map, c_map, s_map

# def msssim_3d_wang03(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
#     weights = tf.convert_to_tensor([0.1333, 0.2363, 0.3001, 0.2856, 0.0448] )

#     img1 = tf.convert_to_tensor( img1 )
#     img2 = tf.convert_to_tensor( img2 )

#     img1 = tf.transpose(img1, [0, 4, 2, 3, 1])
#     img2 = tf.transpose(img2, [0, 4, 2, 3, 1])

#     levels = weights.shape[0]
    
#     ms_ssim = 1

#     for l in range(levels):
#         sim, cs = ssim_exact( tflib.tensor_to_numpy( img1 ), tflib.tensor_to_numpy( img2 ) )

#         l_map, c_map, s_map = ssim_lcs( tflib.tensor_to_numpy( img1 ), tflib.tensor_to_numpy( img2 ) )

#         c_map_w = c_map ** weights[ l ]
#         s_map_w = s_map ** weights[ l ]

#         ms_ssim_l = ms_ssim * ( c_map_w * s_map_w ) 

#         if l == 0:
#             l_map_w = l_map ** weights[ l ]

#             ms_ssim = ms_ssim * l_map_w

#         img1 = tf.nn.avg_pool3d( img1, ksize=2, strides=2, padding="VALID", data_format="NDHWC" )
#         img2 = tf.nn.avg_pool3d( img2, ksize=2, strides=2, padding="VALID", data_format="NDHWC" )

#     return np.mean( ms_ssim )


#----------------------------------------------------------------------------

class MS_SSIM(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
 
        fake_ms_ssim_arr = []


        with tf.device( '/gpu:%d' % 0 ):
            Gs_clone = Gs.clone()
            fake_gen = self._iterate_fakes_norgb( Gs_clone, minibatch_size=1, num_gpus=1 )
            # fake_gen2 = self._iterate_fakes_norgb( Gs_clone, minibatch_size=1, num_gpus=1 )

            for i in range( self.num_images ):
                fake_img1 = next( fake_gen )
                fake_img1 = np.clip( fake_img1, -1.0, 1.0 )
                fake_img1 = np.add( np.divide( fake_img1, 2.0 ), 0.5 )

                fake_img2 = next( fake_gen )
                fake_img2 = np.clip( fake_img2, -1.0, 1.0 )
                fake_img2 = np.add( np.divide( fake_img2, 2.0 ), 0.5 )

                fake_ms_ssim_arr.append( msssim_3d( fake_img1, fake_img2 ) )


        mu_fake = np.mean( fake_ms_ssim_arr )
        sigma_fake = np.std( fake_ms_ssim_arr )

        print( "==============================" )
        print( "mu_fake" )
        print( mu_fake )
        print( "sigma_fake" )
        print( sigma_fake )
        print( "==============================" )
        self._report_result( mu_fake )


        # real_ms_ssim_arr = []

        # img1 = self._iterate_reals_norgb( minibatch_size=1 )
        # img2 = self._iterate_reals_norgb( minibatch_size=1 )


        # for i in range( self.num_images ):
        #     real_ms_ssim_arr.append( msssim_3d( next( img1 ), next( img2 ) ) )

        # del img1
        # del img2

        # mu_real = np.mean( real_ms_ssim_arr )
        # sigma_real = np.std( real_ms_ssim_arr )

        # print( "==============================" )
        # print( "mu_real" )
        # print( mu_real )
        # print( "sigma_real" )
        # print( sigma_real )
        # print( "==============================" )
        # self._report_result( mu_real )


#----------------------------------------------------------------------------
