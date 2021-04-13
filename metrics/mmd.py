# Maximum Mean Discrepancy Distance 
import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

import scipy.ndimage as ndimage

def mmd(img1, img2, B=8):
    """
    maximum mean discrepancy (MMD) based on Gaussian kernel
    function for keras models (theano or tensorflow backend)
    
    - Gretton, Arthur, et al. "A kernel method for the two-sample-problem."
    Advances in neural information processing systems. 2007.
    """
    img1 = img1.reshape( B, -1 )
    img2 = img2.reshape( B, -1 )

    # diff = mix_rbf_mmd2( img1, img2 )

    diff = mmd2_base( img1, img2, B )

    return tflib.tensor_to_numpy( diff )

################################################################################
### Quadratic-time MMD with Gaussian RBF kernel

_eps=1e-8

def mix_rbf_mmd2(X, Y, sigmas=(10,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def mmd2_base(X, Y, B=8):
    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    mmd2 = ( tf.reduce_sum( XX ) + tf.reduce_sum( YY ) - 2 * tf.reduce_sum( XY ) ) / ( B * B ) 
    mmd2 = mmd2

    return mmd2

def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1.0] * len(sigmas)

    XX = tf.matmul(X, X, transpose_b=True)
    XY = tf.matmul(X, Y, transpose_b=True)
    YY = tf.matmul(Y, Y, transpose_b=True)

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = tf.convert_to_tensor( 1.0 / (2.0 * sigma**2.0), dtype=tf.float32 )

        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)


    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
              + tf.reduce_sum(K_YY) / (n * n)
              - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
              + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
              - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2
#----------------------------------------------------------------------------

class MMD(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu

        real_ms_ssim_arr = []
        fake_ms_ssim_arr = []

        train_gen = self._iterate_reals_norgb( minibatch_size=self.minibatch_per_gpu )
        test_gen = self._iterate_test_reals_norgb( minibatch_size=self.minibatch_per_gpu )

        for i in range( self.num_images ):
            real_ms_ssim_arr.append( mmd( next( train_gen ), next( test_gen ), B=self.minibatch_per_gpu ) )


        # with tf.device( '/gpu:%d' % 0 ):
        #     Gs_clone = Gs.clone()
        #     fake_gen = self._iterate_fakes_norgb( Gs_clone, minibatch_size=self.minibatch_per_gpu, num_gpus=1 )

        #     for i in range( self.num_images ):
        #         fake_img = next( fake_gen )
        #         fake_img = np.clip( fake_img, -1.0, 1.0 )
        #         # fake_img = np.add( np.divide( fake_img, 2.0 ), 0.5 )

        #         real_img = next( test_gen )
        #         real_img = np.clip( real_img, 0.0, 1.0 )
        #         real_img = np.multiply( np.add( real_img, -0.5 ), 2 )

        #         # print( "===================================" )
        #         # print( "fake_img : min {} / max {}".format( fake_img.min(), fake_img.max() ) )
        #         # print( "real_img : min {} / max {}".format( real_img.min(), real_img.max() ) )
        #         # print( "===================================" )

        #         fake_ms_ssim_arr.append( mmd( fake_img, real_img, B=self.minibatch_per_gpu ) )


        # mu_fake = np.mean( fake_ms_ssim_arr )
        # sigma_fake = np.std( fake_ms_ssim_arr )

        mu_real = np.mean( real_ms_ssim_arr )
        sigma_real = np.std( real_ms_ssim_arr )

        # print( "==============================" )
        # print( "mu_fake" )
        # print( mu_fake )
        # print( "sigma_fake" )
        # print( sigma_fake )
        # print( "==============================" )

        print( "==============================" )
        print( "mu_real" )
        print( mu_real )
        print( "sigma_real" )
        print( sigma_real )
        print( "==============================" )

        # self._report_result( mu_fake )
        self._report_result( mu_real )

        # # Calculate FID.
        # m = np.square(mu_fake - mu_real).sum()
        # s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False) # pylint: disable=no-member
        # dist = m + np.trace(sigma_fake + sigma_real - 2*s)
        # self._report_result(np.real(dist))

#----------------------------------------------------------------------------
