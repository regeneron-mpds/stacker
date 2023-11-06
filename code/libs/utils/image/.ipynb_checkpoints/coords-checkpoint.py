from PIL import Image
import numpy as np
import neurite as ne
import tensorflow as tf
import pandas as pd
from skimage.draw import disk
from skimage.measure import regionprops_table

def _coords2patch_img(img, coords):
    # pil backend for coords2patch
    upper, left, lower, right = coords
    if not isinstance(img, Image.Image):
        # image assumed to be an array if not already an image
        img = Image.fromarray(img)
    return np.array(img.crop((left, upper, right, lower)))

def _coords2patch_ne(img, coords, out_patch_shape, interp_method='linear',
                     fill_value=None):
  # interpolate arr at the locations defined by upper, left, lower, right
  # have the size specified by out_patch+shape
  # Remember this is rc
  upper, left, lower, right = coords
  coords_ne = np.stack(np.meshgrid(
    np.linspace(upper, lower, out_patch_shape[0]),
    np.linspace(left, right, out_patch_shape[1]),
    indexing='ij'
  ), axis=-1)
  out = ne.utils.interpn(
      tf.convert_to_tensor(img),
      tf.convert_to_tensor(coords_ne),
      interp_method=interp_method,
      fill_value=fill_value).numpy()
  return out

def coords2patch(struct, coords, backend='pil', **kwargs):
    """
    Get a patch from structure `struct` at coordinates `coords`.

    Parameters
    ----------
    struct : np.ndarray
        Structure from which a patch will be extracted.
    coords : length-4 array
        Coordinates for the patch of the form [upper, left, lower, right].
    backend : str
        Backend to use for cropping. Note that backend 'ne' can be used for
        interpolation, but 'pil' is likely faster.
    out_patch_shape : length-2 array
        With backend 'ne': desired size of the output patch.
    interp_method : str
        With backend 'ne': desired interpolation method.
    fill_value : int or None
        With backend 'ne': fill value for out-of-image locations. With
        backend 'pil', the fill value will always be 0.

    Returns
    -------
    np.ndarray
        Extracted patch from image `img` with necessary padding.
    """
    backend = backend.lower()
    if backend == 'pil':
        return _coords2patch_img(struct, coords, **kwargs)
    elif backend == 'ne':
        return _coords2patch_ne(struct, coords, **kwargs)
    else:
        raise ValueError('Backend \'%s\' not in %s.' % (backend, ['pil', 'ne']))

# all args are numpy
def points2lblimg(points, img_shape, point_radius, indexing='ij'):
    # make circles on an image using skimage.draw
    # since numpy, all arguments are ij
    assert indexing.lower() in ('ij', 'xy'), "Indexing must be either 'xy' or 'ij', got '%s'." % indexing.lower()
    assert len(img_shape) > 1 and len(img_shape) < 3, 'Desired image must be 2D with optional third channel dimension.'
    if indexing == 'xy': points = points[..., ::-1]
    img = np.zeros(img_shape, dtype=int)
    for i, point in enumerate(points):
        rr, cc = disk(center=point,
                      radius=point_radius,
                      shape=img.shape[:2])
        img[rr, cc] = i + 1 # one-based index since background is 0
    return img

# assumes all labels present in image
# one-indexed labels
def lblimg2points(img, insert_missing=False, num_points=None, fill_value=np.nan, indexing='ij'):
    assert indexing.lower() in ('ij', 'xy'), "Indexing must be either 'xy' or 'ij', got '%s'." % indexing.lower()
    rprops = pd.DataFrame(regionprops_table(label_image=img,
                                            properties=('label', 'centroid'))).sort_values(by='label')
    # if we need to insert missing points, add -1, -1 dummy points
    if insert_missing:
        max_val = np.amax(rprops['label'].to_numpy())
        if num_points is not None and num_points < max_val: raise ValueError('Expected up to %d labels but got %d.' % (num_points, max_val))
        elif num_points is not None: max_val = num_points
        needs_insertion = np.arange(1, max_val + 1, dtype=int)
        needs_insertion = needs_insertion[~np.isin(needs_insertion, rprops['label'].to_numpy())]
        if needs_insertion.size > 0:
            rprops = pd.concat(objs=[rprops, pd.DataFrame([[lbl, fill_value, fill_value] for lbl in needs_insertion],
                               columns=rprops.columns)],
                               axis=0,
                               ignore_index=True).sort_values(by='label')

    step = 1 if indexing.lower() == 'ij' else -1
    return rprops.loc[:, ['centroid-0', 'centroid-1'][::step]].to_numpy()

def genpatchlocs(img_shape_2d, patch_shape_2d, scale_factor_2d=1, overlap=0):
    # Returns patch locations tiled over an image with img_shape_2d

    # Handle all variables.
    def assertion(var, name):
        if not '__len__' in dir(var): var = np.array((var, var))
        else: var = np.asarray(var)
        assert len(var) == 2, "Length-%d" % len(var) \
            + " %s not compatible with two-dimensional image." % name
        return var
    img_shape_2d, patch_shape_2d, scale_factor_2d, overlap = map(
        assertion,
        (img_shape_2d, patch_shape_2d, scale_factor_2d, overlap),
        ('image shape','patch shape','scale factor', 'overlap')
    )

    # Handle the location information generation.
    starts_big = np.stack((np.meshgrid(
        np.arange(0, img_shape_2d[0]-patch_shape_2d[0]+1, patch_shape_2d[0]-overlap[0]),
        np.arange(0, img_shape_2d[1]-patch_shape_2d[1]+1, patch_shape_2d[1]-overlap[1]),
        indexing='ij'))) # used to do [::-1] but ij should handle it
    ends_big = starts_big + np.array(patch_shape_2d).reshape(-1,1,1)
    # translate to small scale with mult
    loc_info = np.stack((starts_big, ends_big), axis=0) * scale_factor_2d.reshape(1,-1,1,1) 
    loc_info = np.moveaxis(loc_info, source=(0,1), destination=(-2,-1))
    loc_info = loc_info.reshape(*loc_info.shape[:2], -1)
    loc_info = np.rint(loc_info).astype(int) # round off
    return loc_info # organiced into rstart, cstart, rend, cend