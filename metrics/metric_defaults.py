# Please see the license information of the original StyleGAN2 in https://nvlabs.github.io/stylegan2/license.html
# 3D StyleGAN - Metrics

"""Default metric definitions."""

from dnnlib import EasyDict

#----------------------------------------------------------------------------

metric_defaults = EasyDict([(args.name, args) for args in [
    EasyDict(name='mmd_test',    func_name='metrics.mmd.MMD', num_images=10, minibatch_per_gpu=8),
    EasyDict(name='mmd_250',    func_name='metrics.mmd.MMD', num_images=250, minibatch_per_gpu=8),
    EasyDict(name='mmd_1K',    func_name='metrics.mmd.MMD', num_images=1000, minibatch_per_gpu=8),
    EasyDict(name='ms_ssim_test',    func_name='metrics.ms_ssim.MS_SSIM', num_images=5, minibatch_per_gpu=1),
    EasyDict(name='ms_ssim_1K',    func_name='metrics.ms_ssim.MS_SSIM', num_images=1000, minibatch_per_gpu=1),
    EasyDict(name='ms_ssim_2K',    func_name='metrics.ms_ssim.MS_SSIM', num_images=2000, minibatch_per_gpu=1),
    EasyDict(name='ms_ssim_5K',    func_name='metrics.ms_ssim.MS_SSIM', num_images=5000, minibatch_per_gpu=1),
    EasyDict(name='fid_test250',    func_name='metrics.frechet_inception_distance_test.FID_test', num_images=250, minibatch_per_gpu=8),
    EasyDict(name='fid_test1K',    func_name='metrics.frechet_inception_distance_test.FID_test', num_images=1000, minibatch_per_gpu=8),
    EasyDict(name='fid_test2K',    func_name='metrics.frechet_inception_distance_test.FID_test', num_images=2000, minibatch_per_gpu=8),
    EasyDict(name='fid_test5K',    func_name='metrics.frechet_inception_distance_test.FID_test', num_images=5000, minibatch_per_gpu=8),
    EasyDict(name='fid_test4K',    func_name='metrics.frechet_inception_distance_test.FID_test', num_images=4000, minibatch_per_gpu=8),
]])

#----------------------------------------------------------------------------
