import ants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import warnings
import tensorflow as tf

def plt2img(fig):
    # Rasterize the figure
    canvas = FigureCanvas(fig)
    canvas.draw()
    width, height = map(int, fig.get_size_inches() * fig.get_dpi())
    return np.frombuffer(canvas.tostring_rgb(),
                         dtype='uint8').reshape(height, width, 3)

def pil2ants(pil):
    assert pil.mode in ('1', 'L', 'RGB'), 'Unsupported PIL image mode \'%s\'.' % pil.mode
    np_arr = np.asarray(pil).astype(float)
    return ants.from_numpy(np_arr, is_rgb=(np_arr.ndim == 3 and np_arr.shape[-1] == 3))

def def2img(deff1, downsample=1, fig_args={}, quiver_args={}):
    # Make the deformation field
    fig = Figure(**fig_args)
    ax = fig.gca()
    X, Y = np.indices(deff1.shape[:-1])
    U, V = deff1[..., 0], deff1[..., 1]

    # Get the resolution to sample by
    if downsample != 1:
        sample_by_x = deff1.shape[0] // downsample
        sample_by_y = deff1.shape[1] // downsample
        if not sample_by_x:
            warnings.warn('downsample exceeds maximum possible'
                ' in x-direction, setting spacing to 1. Try setting quiver_axis_resolution'
                ' to None.')
            sample_by_x = 1
        if not sample_by_y:
            warnings.warn('downsample exceeds maximum possible'
                ' in y-direction, setting spacing to 1. Try setting quiver_axis_resolution'
                ' to None.')
            sample_by_y = 1
    else: sample_by_x, sample_by_y = 1, 1

    # Make a quiver plot, for some reason ax.axis('square') eliminates all data
    C = quiver_args.pop('C', None)
    if C is not None: ax.quiver(X[::sample_by_x, ::sample_by_y],
                                Y[::sample_by_x, ::sample_by_y],
                                U[::sample_by_x, ::sample_by_y],
                                V[::sample_by_x, ::sample_by_y],
                                C[::sample_by_x, ::sample_by_y],
                                **quiver_args)
    else: ax.quiver(X[::sample_by_x, ::sample_by_y],
                    Y[::sample_by_x, ::sample_by_y],
                    U[::sample_by_x, ::sample_by_y],
                    V[::sample_by_x, ::sample_by_y],
                    **quiver_args)
    
    ax.invert_yaxis()
    ax.axis('equal')
    plt.tight_layout()

    # Rasterize the figure
    return tf.convert_to_tensor(plt2img(fig), dtype=tf.uint8)