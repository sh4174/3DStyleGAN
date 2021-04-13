# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

"""Custom TensorFlow ops for efficient resampling of 2D images."""

import os
import numpy as np
import tensorflow as tf
from .. import custom_ops

def _get_plugin():
    return custom_ops.get_plugin(os.path.splitext(__file__)[0] + '.cu')

#----------------------------------------------------------------------------

def upfirdn_3d(x, k, upx=1, upy=1, upz=1, downx=1, downy=1, downz=1, padx0=0, padx1=0, pady0=0, pady1=0, padz0=0, padz1=0, impl='ref'):
    r"""Pad, upsample, FIR filter, and downsample a batch of 3D images.

    Accepts a batch of 3D images of the shape `[majorDim, inH, inW, inD, minorDim]`
    and performs the following operations for each image, batched across
    `majorDim` and `minorDim`:

    1. Pad the image with zeros by the specified number of pixels on each side
       (`padx0`, `padx1`, `pady0`, `pady1`, `padz0`, `padz1`). Specifying a negative value
       corresponds to cropping the image.

    2. Upsample the image by inserting the zeros after each pixel (`upx`, `upy`, `upz`).

    3. Convolve the image with the specified 3D FIR filter (`k`), shrinking the
       image so that the footprint of all output pixels lies within the input image.

    4. Downsample the image by throwing away pixels (`downx`, `downy`, `downz`).

    This sequence of operations bears close resemblance to scipy.signal.upfirdn().
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Note : H - z, W - y, D - x

    Args:
        x:      Input tensor of the shape `[majorDim, inH, inW, inD, minorDim]`.
        k:      3D FIR filter of the shape `[firH, firW, firD]`.
        upx:    Integer upsampling factor along the X-axis (default: 1).
        upy:    Integer upsampling factor along the Y-axis (default: 1).
        upz:    Integer upsampling factor along the Z-axis (default: 1).
        downx:  Integer downsampling factor along the X-axis (default: 1).
        downy:  Integer downsampling factor along the Y-axis (default: 1).
        downz:  Integer downsampling factor along the Z-axis (default: 1).
        padx0:  Number of pixels to pad on the smaller x side (default: 0).
        padx1:  Number of pixels to pad on the larger x side (default: 0).
        pady0:  Number of pixels to pad on the lower y side (default: 0).
        pady1:  Number of pixels to pad on the larger y side (default: 0).
        padz0:  Number of pixels to pad on the lower z side (default: 0).
        padz1:  Number of pixels to pad on the larger z side (default: 0).
        impl:   Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the shape `[majorDim, outH, outW, outD, minorDim]`, and same datatype as `x`.
    """

    impl_dict = {
        'ref':  _upfirdn_3d_ref,
        # 'cuda': _upfirdn_3d_cuda,
    }
    return impl_dict[impl](x=x, k=k, upx=upx, upy=upy, upz=upz, downx=downx, downy=downy, downz=downz, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1, padz0=padz0, padz1=padz1)

#----------------------------------------------------------------------------

def _upfirdn_3d_ref(x, k, upx, upy, upz, downx, downy, downz, padx0, padx1, pady0, pady1, padz0, padz1):
    """Slow reference implementation of `upfirdn_3d()` using standard TensorFlow ops."""

    x = tf.convert_to_tensor(x)
    k = np.asarray(k, dtype=np.float32)
    assert x.shape.rank == 5
    inH = x.shape[1].value
    inW = x.shape[2].value
    inD = x.shape[3].value
    minorDim = _shape(x, 4)
    kernelH, kernelW, kernelD = k.shape
    assert inW >= 1 and inH >= 1 and inD >= 1
    assert kernelW >= 1 and kernelH >= 1 and kernelD >= 1
    assert isinstance(upx, int) and isinstance(upy, int) and isinstance(upz, int)
    assert isinstance(downx, int) and isinstance(downy, int) and isinstance(downz, int)
    assert isinstance(padx0, int) and isinstance(padx1, int)
    assert isinstance(pady0, int) and isinstance(pady1, int)
    assert isinstance(padz0, int) and isinstance(padz1, int)


    # print( "upfirdn_3d_ref input x.shape" )
    # print( x.shape )


    # Upsample (insert zeros).
    ### x = tf.pad(x, [[0, 0], [0, upz - 1], [0, upy - 1], [0, upx - 1], [0, 0]])
    # x = tf.reshape(x, [ inD, 1, inH, 1, inW, 1 ])
    # x = tf.pad(x, [ [ 0, 0 ], [0, upz - 1], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1] ])
    # x = tf.reshape(x, [-1, inD * upz, inH * upy, inW * upx, minorDim])
    # print( "upx" )
    # print( upx )
    # print( "upy" )
    # print( upy )
    # print( "upz" )
    # print( upz )

    x = tf.keras.layers.UpSampling3D( size=( upx, upy, upz ) )( x )

    # print( "upfirdn_3d_ref upsampled x.shape" )
    # print( x.shape )

    # Pad (crop if negative).
    # x = tf.pad(x, [[0, 0], [max(padz0, 0), max(padz1, 0)], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
    # x = x[:, max(-padz0, 0) : x.shape[1].value - max(-padz1, 0), max(-pady0, 0) : x.shape[2].value - max(-pady1, 0), max(-padx0, 0) : x.shape[3].value - max(-padx1, 0), :]

    # print( "padx0" )
    # print( padx0 )
    # print( "padx1" )    
    # print( padx1 )
    # print( "pady0" )
    # print( pady0 )
    # print( "pady1" )    
    # print( pady1 )
    # print( "padz0" )
    # print( padz0 )
    # print( "padz1" )    
    # print( padz1 )

    padx0_zp = padx0 if padx0 >= 0 else 0
    pady0_zp = pady0 if pady0 >= 0 else 0
    padz0_zp = padz0 if padz0 >= 0 else 0

    padx1_zp = padx1 if padx1 >= 0 else 0
    pady1_zp = pady1 if pady1 >= 0 else 0
    padz1_zp = padz1 if padz1 >= 0 else 0

    x = tf.keras.layers.ZeroPadding3D( padding=( ( padx0_zp, padx1_zp ), ( pady0_zp, pady1_zp ), ( padz0_zp, padz1_zp ) ) )( x )

    cropx0_c = -padx0 if padx0 < 0 else 0
    cropy0_c = -pady0 if pady0 < 0 else 0
    cropz0_c = -padz0 if padz0 < 0 else 0

    cropx1_c = -padx1 if padx1 < 0 else 0
    cropy1_c = -pady1 if pady1 < 0 else 0
    cropz1_c = -padz1 if padz1 < 0 else 0

    x = tf.keras.layers.Cropping3D( cropping=( ( cropx0_c, cropx1_c ), ( cropy0_c, cropy1_c ), ( cropz0_c, cropz1_c ) ) )( x )

    # # Upsample (insert zeros).
    # x = tf.reshape(x, [-1, inH, 1, inW, 1, minorDim])
    # x = tf.pad(x, [[0, 0], [0, 0], [0, upy - 1], [0, 0], [0, upx - 1], [0, 0]])
    # x = tf.reshape(x, [-1, inH * upy, inW * upx, minorDim])

    # # Pad (crop if negative).
    # x = tf.pad(x, [[0, 0], [max(pady0, 0), max(pady1, 0)], [max(padx0, 0), max(padx1, 0)], [0, 0]])
    # x = x[:, max(-pady0, 0) : x.shape[1].value - max(-pady1, 0), max(-padx0, 0) : x.shape[2].value - max(-padx1, 0), :]

    # print( "upfirdn_3d_ref padded x.shape" )
    # print( x.shape )

    # Convolve with filter.
    x = tf.transpose(x, [0, 4, 1, 2, 3 ])
    # x = tf.reshape(x, [-1, 1, inD * upz + padz0 + padz1, inH * upy + pady0 + pady1, inW * upx + padx0 + padx1])
    w = tf.constant(k[::-1, ::-1, ::-1, np.newaxis, np.newaxis], dtype=x.dtype)

    x = tf.nn.conv3d(x, w, strides=[1,1,1,1,1], padding='VALID', data_format='NCDHW')
    # x = tf.reshape(x, [-1, minorDim, inD * upz + padz0 + padz1 - kernelD + 1, inH * upy + pady0 + pady1 - kernelH + 1, inW * upx + padx0 + padx1 - kernelW + 1])
    x = tf.transpose(x, [0, 2, 3, 4, 1])

    # print( "upfirdn_3d_ref convolved x.shape" )
    # print( x.shape )
    # Downsample (throw away pixels).
    return x[:, ::downz, ::downy, ::downx, :]





# #----------------------------------------------------------------------------
# def _upfirdn_2d_cuda(x, k, upx, upy, downx, downy, padx0, padx1, pady0, pady1):
#     """Fast CUDA implementation of `upfirdn_2d()` using custom ops."""

#     x = tf.convert_to_tensor(x)
#     k = np.asarray(k, dtype=np.float32)
#     majorDim, inH, inW, minorDim = x.shape.as_list()
#     kernelH, kernelW = k.shape
#     assert inW >= 1 and inH >= 1
#     assert kernelW >= 1 and kernelH >= 1
#     assert isinstance(upx, int) and isinstance(upy, int)
#     assert isinstance(downx, int) and isinstance(downy, int)
#     assert isinstance(padx0, int) and isinstance(padx1, int)
#     assert isinstance(pady0, int) and isinstance(pady1, int)

#     outW = (inW * upx + padx0 + padx1 - kernelW) // downx + 1
#     outH = (inH * upy + pady0 + pady1 - kernelH) // downy + 1
#     assert outW >= 1 and outH >= 1

#     kc = tf.constant(k, dtype=x.dtype)
#     gkc = tf.constant(k[::-1, ::-1], dtype=x.dtype)
#     gpadx0 = kernelW - padx0 - 1
#     gpady0 = kernelH - pady0 - 1
#     gpadx1 = inW * upx - outW * downx + padx0 - upx + 1
#     gpady1 = inH * upy - outH * downy + pady0 - upy + 1

#     @tf.custom_gradient
#     def func(x):
#         y = _get_plugin().up_fir_dn2d(x=x, k=kc, upx=upx, upy=upy, downx=downx, downy=downy, padx0=padx0, padx1=padx1, pady0=pady0, pady1=pady1)
#         y.set_shape([majorDim, outH, outW, minorDim])
#         @tf.custom_gradient
#         def grad(dy):
#             dx = _get_plugin().up_fir_dn2d(x=dy, k=gkc, upx=downx, upy=downy, downx=upx, downy=upy, padx0=gpadx0, padx1=gpadx1, pady0=gpady0, pady1=gpady1)
#             dx.set_shape([majorDim, inH, inW, minorDim])
#             return dx, func
#         return y, grad
#     return func(x)

#----------------------------------------------------------------------------

def filter_3d(x, k, gain=1, data_format='NCDHW', impl='ref'):
    r"""Filter a batch of 3D images with the given FIR filter.

    Accepts a batch of 3D images of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`
    and filters each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero.

    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`.
        k:            FIR filter of the shape `[firH, firW, firD ]` or `[firN]` (separable).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCDHW'` or `'NHWDC'` (default: `'NCDHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the same shape and datatype as `x`.
    """

    k = _setup_kernel(k) * gain
    p = k.shape[0] - 1
    return _simple_upfirdn_3d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def upsample_3d(x, k=None, factor=2, gain=1, data_format='NCDHW', impl='ref'):
    r"""Upsample a batch of 3D images with the given filter.

    Accepts a batch of 3D images of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`
    and upsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the upsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`.
        k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCDHW'` or `'NHWDC'` (default: `'NCDHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor, D * factor]` or
        `[N, H * factor, W * factor, D * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = k.shape[0] - factor


    pad0 = (k.shape[0] + factor - 1) // 2
    pad1 = (k.shape[0] - factor) // 2

    # return _simple_upfirdn_3d(x, k, up=factor, pad0=(p+1)//2+factor-1, pad1=p//2, data_format=data_format, impl=impl)
    return _simple_upfirdn_3d(x, k, up=factor, pad0=pad0, pad1=pad1, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def downsample_3d(x, k=None, factor=2, gain=1, data_format='NCDHW', impl='ref'):
    r"""Downsample a batch of 3D images with the given filter.

    Accepts a batch of 2D images of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`
    and downsamples each image with the given filter. The filter is normalized so that
    if the input pixels are constant, they will be scaled by the specified `gain`.
    Pixels outside the image are assumed to be zero, and the filter is padded with
    zeros so that its shape is a multiple of the downsampling factor.

    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`.
        k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCDHW'` or `'NHWDC'` (default: `'NCDHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor, D // factor]` or
        `[N, H // factor, W // factor, D // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * gain
    p = k.shape[0] - factor


    pad0 = (k.shape[0] + factor - 1) // 2
    pad1 = (k.shape[0] - factor) // 2


    # return _simple_upfirdn_3d(x, k, down=factor, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)
    return _simple_upfirdn_3d(x, k, down=factor, pad0=pad0, pad1=pad1, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def upsample_conv_3d(x, w, k=None, factor=2, gain=1, data_format='NCDHW', impl='ref'):
    r"""Fused `upsample_3d()` followed by `tf.nn.conv3d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, filterD, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to nearest-neighbor
                      upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCDHW'` or `'NHWDC'` (default: `'NCDHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the shape `[N, C, H * factor, W * factor, D * factor]` or
        `[N, H * factor, W * factor, D * factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1

    # Check weight shape.
    w = tf.convert_to_tensor(w)
    assert w.shape.rank == 5
    convH = w.shape[0].value
    convW = w.shape[1].value
    convD = w.shape[2].value

    inC = _shape(w, 3)
    outC = _shape(w, 4)
    assert convW == convH and convW == convD

    # Setup filter kernel.
    if k is None:
        k = [1] * factor
    k = _setup_kernel(k) * (gain * (factor ** 2))
    p = (k.shape[0] - factor) - (convW - 1)

    # Determine data dimensions.
    if data_format == 'NCDHW':
        stride = [1, 1, factor, factor, factor]
        output_shape = [_shape(x, 0), outC, (_shape(x, 2) - 1) * factor + convH, (_shape(x, 3) - 1) * factor + convW, (_shape(x, 4) - 1) * factor + convD]
        num_groups = _shape(x, 1) // inC
    else:
        stride = [1, factor, factor, factor, 1]
        output_shape = [_shape(x, 0), (_shape(x, 1) - 1) * factor + convH, (_shape(x, 2) - 1) * factor + convW, (_shape(x, 3) - 1) * factor + convD, outC]
        num_groups = _shape(x, 4) // inC

    # Transpose weights.
    w = tf.reshape(w, [convH, convW, convD, inC, num_groups, -1])
    w = tf.transpose(w[::-1, ::-1], [0, 1, 2, 5, 4, 3])
    w = tf.reshape(w, [convH, convW, convD, -1, num_groups * inC])

    # Execute.
    x = tf.nn.conv3d_transpose(x, w, output_shape=output_shape, strides=stride, padding='VALID', data_format=data_format)

    pad0 = (k.shape[0] + factor - convW) // 2
    pad1 = (k.shape[0] - factor - convW + 3) // 2

    # return _simple_upfirdn_3d(x, k, pad0=(p+1)//2+factor-1, pad1=p//2+1, data_format=data_format, impl=impl)
    return _simple_upfirdn_3d(x, k, pad0=pad0, pad1=pad1, data_format=data_format, impl=impl)

#----------------------------------------------------------------------------

def conv_downsample_3d(x, w, k=None, factor=2, gain=1, data_format='NCDHW', impl='ref'):
    r"""Fused `tf.nn.conv3d()` followed by `downsample_3d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.

    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D, C]`.
        w:            Weight tensor of the shape `[filterH, filterW, filterD, inChannels, outChannels]`.
                      Grouped convolution can be performed by `inChannels = x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]` (separable).
                      The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCDHW'` or `'NHWDC'` (default: `'NCDHW'`).
        impl:         Name of the implementation to use. Can be `"ref"` (default) : (TBD) or `"cuda"`.

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor, D // factor]` or
        `[N, H // factor, W // factor, D // factor, C]`, and same datatype as `x`.
    """

    assert isinstance(factor, int) and factor >= 1
    w = tf.convert_to_tensor(w)
    convH, convW, convD, _inC, _outC = w.shape.as_list()
    assert convW == convH and convW == convD
    if k is None:
        k = [1, 1, 1] * factor
    k = _setup_kernel(k) * gain
    p = (k.shape[0] - factor) + (convW - 1)
    if data_format == 'NCDHW':
        s = [1, 1, factor, factor, factor]
    else:
        s = [1, factor, factor, factor, 1]



    pad0 = (k.shape[0] - factor + convW) // 2
    pad1 = (k.shape[0] - factor + convW - 1) // 2

    # x = _simple_upfirdn_3d(x, k, pad0=(p+1)//2, pad1=p//2, data_format=data_format, impl=impl)
    x = _simple_upfirdn_3d(x, k, pad0=pad0, pad1=pad1, data_format=data_format, impl=impl)

    return tf.nn.conv3d(x, w, strides=s, padding='VALID', data_format=data_format)


#----------------------------------------------------------------------------
# Internal helper funcs.

def _shape(tf_expr, dim_idx):
    if tf_expr.shape.rank is not None:
        dim = tf_expr.shape[dim_idx].value
        if dim is not None:
            return dim
    return tf.shape(tf_expr)[dim_idx]

def _setup_kernel(k):
    k = np.asarray(k, dtype=np.float32)

    if k.ndim == 1:
        k_tdot = np.tensordot(k, k, axes=0)
        k = np.tensordot( k_tdot, k, axes= 0 )

    k = np.divide( k, np.sum(k) )
    assert k.ndim == 3
    assert k.shape[0] == k.shape[1] and k.shape[ 0 ] == k.shape[ 2 ]
    return k

def _simple_upfirdn_3d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCDHW', impl='ref'):
    assert data_format in ['NCDHW', 'NDHWC']
    assert x.shape.rank == 5
    y = x

    
    if data_format == 'NCDHW':
        y = tf.reshape(y, [-1, _shape(y, 2), _shape(y, 3), _shape(y, 4), 1])

    y = upfirdn_3d(y, k, upx=up, upy=up, upz=up, downx=down, downy=down, downz=down, padx0=pad0, padx1=pad1, pady0=pad0, pady1=pad1, padz0=pad0, padz1=pad1, impl=impl)


    if data_format == 'NCDHW':
        y = tf.reshape(y, [-1, _shape(x, 1), _shape(y, 1), _shape(y, 2), _shape(y, 3)])

    return y

#----------------------------------------------------------------------------
