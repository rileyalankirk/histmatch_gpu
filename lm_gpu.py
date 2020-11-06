# GPU implementations of LM functions
import cupy as cp
from cupy import array, empty, linspace, place, uint64, zeros
from cupyx.scipy.ndimage import correlate
from hist.exact import __check_h_dst
from hist.exact.lm import __get_filters
from hist.util import as_unsigned, check_image_mask_single_channel, get_dtype_min_max, FLOAT64_NMANT


def __calc_info_gpu(im, order=6):    
    # Deal with arguments
    if order < 2: raise ValueError('order')
    im = as_unsigned(im)
    dt = im.dtype

    # Get the filters for this setup
    filters, includes_order_one = __get_filters(dt, order, im.ndim)
    filters = [cp.array(_) for _ in filters]

    if len(filters) == 1 and (includes_order_one or FLOAT64_NMANT + dt.itemsize*8 <= 64):
        # Single convolution
        out = correlate(im, filters[0], empty(im.shape, uint64))
        if not includes_order_one:
            out |= im.astype(uint64) << FLOAT64_NMANT
        return out

    # Convolve filters with the image and stack
    im = im.astype(float, copy=False)
    out = empty((len(filters)+(not includes_order_one),) + im.shape)
    for i, fltr in enumerate(filters):
        correlate(im, fltr, out[i, ...])
    if not includes_order_one:
        out[-1, ...] = im
    return out

def __sort_pixels_gpu(values, shape, mask=None, return_fails=False, stable=False):
    ##### Check if the values already contain the failures #####
    fails = None
    if return_fails and isinstance(values, tuple):
        values, fails = values

    ##### Assign strict ordering #####
    if values.ndim == 1:
        # Already sorted
        from hist.util import prod
        assert values.size == prod(shape)
        idx = values
    elif values.shape == shape:
        # Single value per pixel
        values = values.ravel()
        idx = values.argsort()
    else:
        # Tuple of values per pixel - need lexsort
        from cupy import lexsort
        assert values.shape[1:] == shape
        values = values.reshape(values.shape[0], -1)
        idx = lexsort(values)

    # Done if not calculating failures
    if not return_fails or fails is not None: return idx, fails

    # Calculate the number of sort failures
    values = values.T # for lexsorted values
    values_sorted = values[idx]
    not_equals = values_sorted[1:] != values_sorted[:-1]
    del values_sorted
    if not_equals.ndim == 2: not_equals = not_equals.any(1) # for lexsorted values
    return idx, int(not_equals.size - not_equals.sum())

def __calc_transform_gpu(h_dst, dt, n_mask, n_full):
    mn, mx = get_dtype_min_max(dt)
    transform = zeros(n_full, dtype=dt)
    transform[-n_mask:] = linspace(mn, mx, len(h_dst), dtype=dt).repeat(h_dst.tolist())
    return transform

def __apply_transform_gpu(idx, transform, shape, mask=None):
    out = empty(shape, transform.dtype)
    out.put(idx, transform)
    return out

def histeq_exact_gpu(im, h_dst=256, mask=None, method='lm', return_fails=False, stable=False, **kwargs): #pylint: disable=too-many-arguments
    # Check arguments
    im, mask = check_image_mask_single_channel(im, mask)
    n = im.size
    h_dst = array(__check_h_dst(h_dst, n))

    ##### Create strict-orderable versions of image #####
    # These are frequently floating-point 'images' and/or images with an extra dimension giving a
    # 'tuple' of data for each pixel
    values = __calc_info_gpu(im, **kwargs)

    ##### Assign strict ordering #####
    idx, fails = __sort_pixels_gpu(values, im.shape, mask, return_fails, stable)
    del values

    ##### Create the transform that is the size of the image but with sorted histogram values #####
    transform = __calc_transform_gpu(h_dst, im.dtype, n, idx.size)
    del h_dst

    ##### Create the equalized image #####
    out = __apply_transform_gpu(idx, transform, im.shape, mask)

    # Done
    return (out, fails) if return_fails else out
