# Please see the license information of the original StyleGAN2 in https://nvlabs.github.io/stylegan2/license.html

"""Frechet Inception Distance (FID). - Slice-wise"""

import os
import numpy as np
import scipy
import tensorflow as tf
import dnnlib.tflib as tflib

from metrics import metric_base
from training import misc

from PIL import Image as p_img
#----------------------------------------------------------------------------

class FID_test(metric_base.MetricBase):
    def __init__(self, num_images, minibatch_per_gpu, **kwargs):
        super().__init__(**kwargs)
        self.num_images = num_images
        self.minibatch_per_gpu = minibatch_per_gpu

        # self.num_images=3

    def _evaluate(self, Gs, Gs_kwargs, num_gpus):
        minibatch_size = num_gpus * self.minibatch_per_gpu
        inception = misc.load_pkl('http://d36zk2xti64re0.cloudfront.net/stylegan1/networks/metrics/inception_v3_features.pkl')

        print( "Inception Size" )
        print( inception.output_shape[1] )
        activations = np.empty([self.num_images, inception.output_shape[1]], dtype=np.float32)

        # # Calculate statistics for reals.
        # cache_file = self._get_cache_file_for_reals(num_images=self.num_images)

        # os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        # if os.path.isfile(cache_file):
        #     mu_real, sigma_real = misc.load_pkl(cache_file)
        # else:
        #     for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
        #         begin = idx * minibatch_size
        #         end = min(begin + minibatch_size, self.num_images)

        #         print( "=========================================" )
        #         print( "Real Images" )
        #         print( images.shape )
        #         print( images[ 0, 0, images.shape[2]//2, images.shape[3]//2, images.shape[4]//2 ] )
        #         print( images.max() )
        #         print( images.min() )
        #         print( "=========================================" )

        #         # images = tflib.convert_3d_images_to_uint8(images, drange=[ 0, 1 ] )

        #         # 2D middle slice for 3D Images
        #         if len( images.shape ) == 5:
        #             images = images[ :, :, :, :, images.shape[4]//2 ]

        #         activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)

        #         if end == self.num_images:
        #             break

        #     mu_real = np.mean(activations, axis=0)
        #     sigma_real = np.cov(activations, rowvar=False)

        #     print( "==============================" )
        #     print( "mu_real" )
        #     print( mu_real )
        #     print( "sigma_real" )
        #     print( sigma_real )
        #     print( "==============================" )
        #     misc.save_pkl((mu_real, sigma_real), cache_file)


        for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)

            print( "=========================================" )
            print( "Real Images" )
            print( images.shape )
            print( images[ 0, 0, images.shape[2]//2, images.shape[3]//2, images.shape[4]//2 ] )
            print( images.max() )
            print( images.min() )
            print( "=========================================" )

            # images = tflib.convert_3d_images_to_uint8(images, drange=[ 0, 1 ] )

            # 2D middle slice for 3D Images
            if len( images.shape ) == 5:
                # images = images[ :, :, images.shape[2]//2, :, : ]
                # images = images[ :, :, :, images.shape[3]//2, : ]
                images = images[ :, :, :, :, images.shape[4]//2 ]


            activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)

            if end == self.num_images:
                break

        mu_real = np.mean(activations, axis=0)
        sigma_real = np.cov(activations, rowvar=False)

        # Second real from the same datasets
        for idx, images in enumerate(self._iterate_reals(minibatch_size=minibatch_size)):
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)

            print( "=========================================" )
            print( "Real Images" )
            print( images.shape )
            print( images[ 0, 0, images.shape[2]//2, images.shape[3]//2, images.shape[4]//2 ] )
            print( images.max() )
            print( images.min() )
            print( "=========================================" )

            # images = tflib.convert_3d_images_to_uint8(images, drange=[ 0, 1 ] )

            # 2D middle slice for 3D Images
            if len( images.shape ) == 5:
                # images = images[ :, :, images.shape[2]//2, :, : ]
                # images = images[ :, :, :, images.shape[3]//2, : ]
                images = images[ :, :, :, :, images.shape[4]//2 ]

            activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)

            if end == self.num_images:
                break

        mu_real2 = np.mean(activations, axis=0)
        sigma_real2 = np.cov(activations, rowvar=False)

        print( "==============================" )
        print( "mu_real" )
        print( mu_real2 )
        print( "sigma_real" )
        print( sigma_real2 )
        print( "==============================" )

        for idx, images in enumerate(self._iterate_test_reals(minibatch_size=minibatch_size)):
            begin = idx * minibatch_size
            end = min(begin + minibatch_size, self.num_images)

            print( "=========================================" )
            print( "Real Images" )
            print( images.shape )
            print( images[ 0, 0, images.shape[2]//2, images.shape[3]//2, images.shape[4]//2 ] )
            print( images.max() )
            print( images.min() )
            print( "=========================================" )

            # images = tflib.convert_3d_images_to_uint8(images, drange=[ 0, 1 ] )

            # 2D middle slice for 3D Images
            if len( images.shape ) == 5:
                # images = images[ :, :, images.shape[2]//2, :, : ]
                # images = images[ :, :, :, images.shape[3]//2, : ]
                images = images[ :, :, :, :, images.shape[4]//2 ]

            activations[begin:end] = inception.run(images[:end-begin], num_gpus=num_gpus, assume_frozen=True)

            if end == self.num_images:
                break

        mu_test_real = np.mean(activations, axis=0)
        sigma_test_real = np.cov(activations, rowvar=False)

        print( "==============================" )
        print( "mu_test_real" )
        print( mu_test_real )
        print( "sigma_test_real" )
        print( sigma_test_real )
        print( "==============================" )

        # Construct TensorFlow graph.
        result_expr = []
        for gpu_idx in range(num_gpus):
            with tf.device('/gpu:%d' % gpu_idx):
                Gs_clone = Gs.clone()
                inception_clone = inception.clone()
                latents = tf.random_normal([self.minibatch_per_gpu] + Gs_clone.input_shape[1:])
                labels = self._get_random_labels_tf(self.minibatch_per_gpu)
                images = Gs_clone.get_output_for(latents, None, **Gs_kwargs)

                # images = ( images + 1.0 ) / 2.0 * 255.0 + 0.5

                # 2D middle slice for 3D Images
                if len( images.shape ) == 5:
                    # images = images[ :, :, images.shape[2]//2, :, : ]
                    # images = images[ :, :, :, images.shape[3]//2, : ]
                    images = images[ :, :, :, :, images.shape[4]//2 ]

                images = tflib.convert_images_to_uint8( images )

                print('shape before', images.shape)
                if images.shape[1] == 1:
                    #images = tf.repeat(images, 3, axis=1)
                    images = tf.concat([images, images, images], axis=1)
                    #images = tf.stack([images, images, images], axis=1)

                # images = tf.transpose(images, [0, 2, 3, 1 ])
                # images = tf.image.resize( images, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR )
                # images = tf.transpose(images, [ 0, 3, 1, 2 ] )

                print('shape expanded ', images.shape)
                print( np.max( images.eval() ) ) 
                print( np.min( images.eval() ) ) 
                result_expr.append(inception_clone.get_output_for(images))

        # Calculate statistics for fakes.
        for begin in range(0, self.num_images, minibatch_size):
            print( "begin" )
            print( begin )
            self._report_progress(begin, self.num_images)
            end = min(begin + minibatch_size, self.num_images)
            activations[begin:end] = np.concatenate(tflib.run(result_expr), axis=0)[:end-begin]
        
        mu_fake = np.mean(activations, axis=0)
        sigma_fake = np.cov(activations, rowvar=False)

        print( "==============================" )
        print( "mu_fake" )
        print( mu_fake )
        print( "sigma_fake" )
        print( sigma_fake )
        print( "==============================" )

        # Calculate FID - fake 
        m = np.square(mu_fake - mu_test_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_test_real), disp=False) # pylint: disable=no-member

        print( "fake s" )
        print( s ) 
        print( "fake trace sigma /s" )
        print( np.real( np.trace(sigma_fake + sigma_test_real - 2*s) ) )
        print( "fake mean diff" )
        print( m )

        dist = m + np.trace(sigma_fake + sigma_test_real - 2*s)
        self._report_result(np.real(dist))

        print( "==========================================" )
        print( "FID Fake Images / Test Real Images" )
        print( np.real( dist ) ) 
        print( "==========================================" )

        # Calculate FID - training real 
        m = np.square(mu_real - mu_test_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_real, sigma_test_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_real + sigma_test_real - 2*s)
        self._report_result(np.real(dist))

        print( "training real/ test real s" )
        print( s ) 
        print( "training real/ test real trace sigma /s" )
        print( np.real( np.trace(sigma_fake + sigma_test_real - 2*s) ) )
        print( "training real/ test real mean diff" )
        print( m )
        

        print( "==========================================" )
        print( "FID Training Real Images / Test Real Images" )
        print( np.real( dist ) ) 
        print( "==========================================" )

        # Calculate FID - training real / training
        m = np.square(mu_real2 - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_real2, sigma_real), disp=False) # pylint: disable=no-member
        dist = m + np.trace(sigma_real2 + sigma_real - 2*s)
        self._report_result(np.real(dist))

        print( "==========================================" )
        print( "FID Training Real Images / Training Real Images" )
        print( np.real( dist ) ) 
        print( "==========================================" )


#----------------------------------------------------------------------------
