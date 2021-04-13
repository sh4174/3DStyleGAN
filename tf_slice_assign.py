import numpy as np
# NOTE: numpy is imported for argsorting. We might not use it but then lose in
# code clarity (and a bit in speed but negligible).
import tensorflow as tf


def slice_assign(sliced_tensor, assigned_tensor, *slice_args, verbose=0):
    """Assign a tensor to the slice of another tensor.

    No broadcast is performed.

    Args:
        - sliced_tensor (tf.Tensor): the tensor whose slice you want changed.
        - assigned_tensor (tf.Tensor): the tensor which you want assigned.
        - *slice_args (str or slice): the slices arguments. Can be ':', '...'
        or slice.

    Returns:
        - tf.Tensor: the original tensor with the slice correctly assigned.
    """
    #shape = tf.shape(sliced_tensor)
    shape = sliced_tensor.shape
    print(shape)
    n_dims = len(shape)
    #n_dims = shape.shape
    # parsing the slice specifications
    n_slices = len(slice_args)
    dims_to_index = []
    corresponding_ranges = []
    ellipsis = False
    for i_dim, slice_spec in enumerate(slice_args):
        if isinstance(slice_spec, str):
            if slice_spec == ':':
                continue
            elif slice_spec == '...':
                ellipsis = True
            else:
                raise ValueError('Slices must be :, ..., or slice object.')
        elif slice_spec is Ellipsis:
            ellipsis = True
        else:
            start, stop, step = slice_spec.start, slice_spec.stop, slice_spec.step
            no_start = start is None or start == 0
            no_stop = stop is None or stop == -1
            no_step = step is None or step == 1
            if no_start and no_stop and no_step:
                continue
            if ellipsis:
                real_index = i_dim + (n_dims - n_slices)
            else:
                real_index = i_dim
            dims_to_index.append(real_index)
            if no_step:
                step = 1
            if no_stop:
                stop = shape[real_index]
            if no_start:
                start = 0
            corresponding_range = tf.range(start, stop, step)
            corresponding_ranges.append(corresponding_range)
    if not dims_to_index:
        if verbose > 0:
            print('Warning: no slicing performed')
        return assigned_tensor
    dims_left_out = [
        i_dim for i_dim in range(n_dims) if i_dim not in dims_to_index
    ]
    scatted_nd_perm = dims_to_index + dims_left_out
    inverse_scatter_nd_perm = list(np.argsort(scatted_nd_perm))
    # reshaping the tensors
    # NOTE: the tensors are reshaped to allow for easier indexing with
    # tensor_scatter_nd_update
    sliced_tensor_reshaped = tf.transpose(sliced_tensor, perm=scatted_nd_perm)
    assigned_tensor_reshaped = tf.transpose(assigned_tensor, perm=scatted_nd_perm)
    left_out_shape = [shape[i_dim] for i_dim in dims_left_out]
    assigned_tensor_reshaped = tf.reshape(assigned_tensor_reshaped, [-1] + left_out_shape)
    # creating the indices
    mesh_ranges = tf.meshgrid(*corresponding_ranges, indexing='ij')
    update_indices = tf.stack([
        tf.reshape(slicing_range, (-1,))
        for slicing_range in mesh_ranges
    ], axis=-1)

    # finalisation
    sliced_tensor_reshaped = tf.tensor_scatter_nd_update(
        tensor=sliced_tensor_reshaped,
        indices=update_indices,
        updates=assigned_tensor_reshaped,
    )
    sliced_tensor_updated = tf.transpose(
        sliced_tensor_reshaped,
        perm=inverse_scatter_nd_perm,
    )
    return sliced_tensor_updated
