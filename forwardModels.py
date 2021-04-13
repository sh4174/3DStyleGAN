from abc import ABC, abstractmethod

import tensorflow as tf
import dnnlib
import dnnlib.tflib as tflib
from dnnlib.tflib.ops.upfirdn_2d import upsample_2d, downsample_2d, upsample_conv_2d, conv_downsample_2d
from dnnlib.tflib.ops.fused_bias_act import fused_bias_act

from tf_slice_assign import slice_assign

import cv2
import numpy as np
import pdb
import scipy.ndimage.morphology

class ForwardAbstract(ABC):
  def __init__(self):
    pass

  @abstractmethod
  def __call__(self, x):
    return x

  def calcMaskFromImg(self, img):
    pass
  
  def initVars(self):
    pass
  
  def getVars(self):
    return []

class ForwardNone(ForwardAbstract):
  def __init__(self):
    pass

  def __call__(self, x):
    return x

class ForwardDownsample(ForwardAbstract):
  def __init__(self, res):
    self.res = res # target resolution

  # resolution of input x can be anything, but aspect ratio should be 1:1
  def __call__(self, x): 
    factor = int(x.shape[2] // self.res)
    # ensure it performs bicubic downsampling
    #return downsample_2d(x, factor=factor, data_format='NCHW')
    x = tf.reshape(x, (x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
    x = tf.image.resize(x, [self.res, self.res], method='bicubic') # BHWC format 
    x = tf.reshape(x, (x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
    return x

class ForwardFillBoundingBox(ForwardAbstract):
  """ Takes a bounding box and fills it with zero. Modelled after tf.image.crop_to_bounding_box. """
  def __init__(self, offset_height, offset_width, target_height, target_width, opt_params=True):
    self.offset_height = offset_height
    self.offset_width = offset_width
    self.target_height = target_height
    self.target_width = target_width

    if opt_params:
      self.startH = tf.Variable(self.offset_height, name = 'startH', trainable=True, dtype=tf.float32)
      self.endH = tf.Variable(self.offset_height + self.target_height, name = 'endH', trainable=True, dtype=tf.float32)
      self.startW = tf.Variable(self.offset_width, name = 'startW', trainable=True, dtype=tf.float32)
      self.endW = tf.Variable(self.offset_width + self.target_width, name='endW', trainable=True)
#self.startH = self.offset_height
    #self.endH = self.offset_height + self.target_height
    #self.startW = self.offset_width
    #self.endW = self.offset_width + self.target_width
    else:
      self.startH = self.offset_height
      self.endH = self.offset_height + self.target_height
      self.startW = self.offset_width
      self.endW = self.offset_width + self.target_width

    #self.vars = [self.startH, self.endH, self.startW, sels.endW]
    self.initVals = [self.offset_height, self.offset_height + self.target_height, self.offset_width, self.offset_width + self.target_width]
 
  # resolution of input x can be anything, but aspect ratio should be 1:1
  def __call__(self, x): 
   
    #x = tf.Variable(x)
    #x[:,:,startH : endH, startW : endW].assign(tf.zeros([endH-startH,endW-startW]))

    fill = tf.zeros((x.shape[0], x.shape[1], self.target_height, self.target_width))
    xfill = slice_assign(x, fill, ':', ':', slice(self.startH, self.endH, 1), slice(self.startW, self.endW, 1))

    return xfill

  def getVars(self):
    return [self.startH, self.endH, self.startW, self.endW]

  def initVars(self):
    tflib.set_vars({
      self.startH : self.initVals[0],
      self.endH : self.initVals[1],
      self.startW : self.initVals[2],
      self.endH : self.initVals[3]
    })

class ForwardFillMask(ForwardAbstract):
  """ Takes an image with a filled-in mask (already baked in the image), and derived the mask automatically by taking the histogram over voxels. Supports free-form masks """
  def __init__(self):
    self.mask = None

  def calcMaskFromImg(self, img):
    print('before hist')
    print(img.shape)
    #pdb.set_trace()
    nrBins = 256
    grayImg = np.mean(np.squeeze(img), axis=0)
    hist,bins = np.histogram(grayImg.ravel(), nrBins, [-1,1])
    #hist = cv2.calcHist(np.squeeze(img[0,:,:,:]), channels=[0], mask=None, histSize=[nrBins], ranges=[-1,1])
    print(hist, bins)
    maxIndex = np.argmax(hist)

    self.mask = np.abs(grayImg - bins[maxIndex]) < (2.0/nrBins)
    self.mask = scipy.ndimage.morphology.binary_opening(self.mask, iterations=3)

    print('nr True', np.sum(self.mask))
    print('nr False', np.sum(~self.mask))
    self.mask = tf.repeat(tf.reshape(self.mask, (1, 1, *self.mask.shape)), img.shape[1], axis=1)

  
  def __call__(self, x): 
    #if (self.mask is None) and tf.is_tensor(x):
    #  self.calcMaskFromImg(x)
    if (self.mask is None):
      self.mask = tf.zeros(x.shape, dtype=bool)

    print('__call__', self.mask.shape)
    zeroFill = tf.zeros(x.shape)
    xFill = tf.where(self.mask, zeroFill, x) # if true, then zeroFill, else x

    return xFill

class ForwardBlurWithKernel(ForwardAbstract):
  def __init__(self):
    self.kernel = None


  def __call__(self, x):

    return x

class ForwardUndersampleMRI(ForwardAbstract):
  def __init__(self, img_size, corruption_frac):
    #img_size = images.shape[1]
    self.num_points = int(img_size * corruption_frac)
    self.coord_list = np.random.choice( # indices of lines to corrupt in the k-space
        range(img_size), self.num_points, replace=False)

    self.mask = np.ones((1, img_size, img_size, 2)).astype(bool)
    for k in range(len(self.coord_list)):
        self.mask[0, self.coord_list[k], :, :] = False
    #self.mask[0, 64:192,:,:] = False


  def __call__(self, x):
    x_gray = tf.reduce_mean(x,axis=1, keepdims=False) # convert to grayscale as we still didn't move to 1-channel StyleGAN 
    corrupt_x, _ = self.undersample_image(x_gray)
    
    # taking the magnitude results in sign-flip, which inverts the image if the input is in the negatives. Brains&Xrays images all have negative values, which is strange
    #corrupt_x = tf.sqrt(corrupt_x[:,:,:,0]**2 + corrupt_x[:,:,:,1]**2) #take the magnitude of the complex tensor returned
    corrupt_x = corrupt_x[:,:,:,0]
    return tf.repeat(tf.reshape(corrupt_x, (1,1,corrupt_x.shape[1], -1)), repeats=3, axis=1)

#  taken from Singh et al: https://github.com/nalinimsingh/interlacer/blob/master/interlacer/data_generator.py#L106
  def undersample_image(self, 
    images
    ):
    """Generator that yields batches of undersampled input and correct output data.
    For corrupted inputs, select each line in k-space with probability corruption_frac and set it to zero.
    Args:
      images(float): Numpy array of input images, of shape (num_images, n, n)
      input_domain(str): The domain of the network input; 'FREQ' or 'IMAGE'
      output_domain(str): The domain of the network output; 'FREQ' or 'IMAGE'
      corruption_frac(float): Probability with which to zero a line in k-space
      batch_size(int, optional): Number of input-output pairs in each batch (Default value = 10)
    Returns:
      inputs: Tuple of corrupted input data and ground truth output data, both numpy arrays of shape (batch_size,n,n,2).
    """

    images = split_reim_tensor(images)
    true_k = convert_tensor_to_frequency_domain(images)

    zeroFill = tf.zeros(true_k.shape)
    corrupt_k = tf.where(self.mask, true_k, zeroFill) # if true, then true_k, else zeroFill
    print(self.mask.shape)
    print(true_k.shape)
    print(zeroFill.shape)
    #corrupt_k = true_k

    corrupt_img = convert_tensor_to_image_domain(corrupt_k)
    return corrupt_img, corrupt_k

  def get_k(self, images):
    images = split_reim_tensor(images)
    true_k = convert_tensor_to_frequency_domain(images)
    zeroFill = tf.zeros(true_k.shape)
    corrupt_k = tf.where(self.mask, true_k, zeroFill) # if true, then true_k, else zeroFill
    return true_k, corrupt_k
    

def split_reim(array):
    """Split a complex valued matrix into its real and imaginary parts.
    Args:
      array(complex): An array of shape (batch_size, N, N) or (batch_size, N, N, 1)
    Returns:
      split_array(float): An array of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel
    """
    real = tf.math.real(array)
    imag = tf.math.imag(array)
    split_array = tf.stack([real, imag], axis=3)
    return split_array


def split_reim_tensor(array):
    """Split a complex valued tensor into its real and imaginary parts.
    Args:
      array(complex): A tensor of shape (batch_size, N, N) or (batch_size, N, N, 1)
    Returns:
      split_array(float): A tensor of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel
    """
    real = tf.math.real(array)
    imag = tf.math.imag(array)
    split_array = tf.stack((real, imag), axis=3)
    return split_array


def split_reim_channels(array):
    """Split a complex valued tensor into its real and imaginary parts.
    Args:
      array(complex): A tensor of shape (batch_size, N, N) or (batch_size, N, N, 1)
    Returns:
      split_array(float): A tensor of shape (batch_size, N, N, 2) containing the real part on one channel and the imaginary part on another channel
    """
    real = tf.math.real(array)
    imag = tf.math.imag(array)
    n_ch = array.get_shape().as_list()[3]
    split_array = tf.concat((real, imag), axis=3)
    return split_array


def join_reim(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.
    Args:
      array(float): An array of shape (batch_size, N, N, 2)
    Returns:
      joined_array(complex): An complex-valued array of shape (batch_size, N, N, 1)
    """
    print('type array', type(array))
    #joined_array = array[:, :, :, 0] + 1j * array[:, :, :, 1]
    joined_array = tf.complex(array[:, :, :, 0],  array[:, :, :, 1])
    return joined_array


def join_reim_tensor(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.
    Args:
      array(float): An array of shape (batch_size, N, N, 2)
    Returns:
      joined_array(complex): A complex-valued array of shape (batch_size, N, N)
    """
    joined_array = tf.cast(array[:, :, :, 0], 'complex64') + \
        1j * tf.cast(array[:, :, :, 1], 'complex64')
    return joined_array


def join_reim_channels(array):
    """Join the real and imaginary channels of a matrix to a single complex-valued matrix.
    Args:
      array(float): An array of shape (batch_size, N, N, ch)
    Returns:
      joined_array(complex): A complex-valued array of shape (batch_size, N, N, ch/2)
    """
    ch = array.get_shape().as_list()[3]
    joined_array = tf.cast(array[:,
                                 :,
                                 :,
                                 :int(ch / 2)],
                           dtype=tf.complex64) + 1j * tf.cast(array[:,
                                                                    :,
                                                                    :,
                                                                    int(ch / 2):],
                                                              dtype=tf.complex64)
    return joined_array


def convert_to_frequency_domain(images):
    """Convert an array of images to their Fourier transforms.
    Args:
      images(float): An array of shape (batch_size, N, N, 2)
    Returns:
      spectra(float): An FFT-ed array of shape (batch_size, N, N, 2)
    """
    n = images.shape[1]
    print(images.shape)
    asd
    #spectra = split_reim(np.fft.fft2(join_reim(images), axes=(1, 2)))
    spectra = split_reim(tf.signal.fft2d(join_reim(images)))
    return spectra


def convert_tensor_to_frequency_domain(images):
    """Convert a tensor of images to their Fourier transforms.
    Args:
      images(float): A tensor of shape (batch_size, N, N, 2)
    Returns:
      spectra(float): An FFT-ed tensor of shape (batch_size, N, N, 2)
    """
    n = images.shape[1]
    spectra = split_reim_tensor(tf.signal.fft2d(join_reim_tensor(images)))
    return spectra


def convert_to_image_domain(spectra):
    """Convert an array of Fourier spectra to the corresponding images.
    Args:
      spectra(float): An array of shape (batch_size, N, N, 2)
    Returns:
      images(float): An IFFT-ed array of shape (batch_size, N, N, 2)
    """
    n = spectra.shape[1]
    #images = split_reim(np.fft.ifft2(join_reim(spectra), axes=(1, 2)))
    images = split_reim(tf.signal.ifft2d(join_reim(spectra)))
    return images


def convert_tensor_to_image_domain(spectra):
    """Convert an array of Fourier spectra to the corresponding images.
    Args:
      spectra(float): An array of shape (batch_size, N, N, 2)
    Returns:
      images(float): An IFFT-ed array of shape (batch_size, N, N, 2)
    """
    n = spectra.shape[1]
    images = split_reim_tensor(tf.signal.ifft2d(join_reim_tensor(spectra)))
    return images
