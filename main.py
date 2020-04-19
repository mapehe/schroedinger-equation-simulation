# Import libraries for simulation
import tensorflow.compat.v1 as tf
import numpy as np
import math
import itertools

# Imports for visualization
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display
import colorsys


def render(a, image_name, brightness=0.01):
    """Display an array as a picture."""
    def z2rgb(z):
        def hsv2rgb(h, s, v):
            return tuple(int(round(i * 255))
                         for i in colorsys.hls_to_rgb(h, s, v))

        def z2hsv(x, y):
            arg = math.atan2(y, x)
            nrm = math.sqrt(x**2 + y**2)
            return (arg, 1 - brightness**(nrm), 1)
        return hsv2rgb(*z2hsv(*z))

    state = a[0, :, :, :]
    xmax, ymax, _ = state.shape
    color_array = np.zeros((xmax, ymax, 3), 'uint8')
    for i in range(xmax):
        for j in range(ymax):
            color_array[i, j, :] = np.array(z2rgb(state[i, j, :]))
    PIL.Image.fromarray(color_array).save(image_name)


def complex_mul(z1, z2):
    """Multiply two complex tensors."""
    x1, y1 = tf.split(z1, [1, 1], -1)
    x2, y2 = tf.split(z2, [1, 1], -1)
    return tf.concat((
        tf.add(tf.math.multiply(x1, x2), -1 * tf.math.multiply(y1, y2)),
        tf.add(tf.math.multiply(x1, y2), tf.math.multiply(x2, y1))), -1)

def complex_scale(a, z):
    """Scale a complex tensor z by a."""
    x1, y1 = tf.split(z, [1, 1], -1)
    x2, y2 = a[0], a[1]
    return tf.concat((x2 * x1 - y1 * y2,
        x1 * y2 + x2 * y1), -1)


def laplace(x):
    """Compute the laplacian"""
    def laplacian_filter(d):
        def index_to_value(a):
            """
                Compute the value corresponding to a multi-index a. For further info see
                "Implementation via operator discretization" in
                https://en.wikipedia.org/wiki/Discrete_Laplace_operator
            """
            count = sum(map(lambda x: abs(x - 1), a))
            if count == 0:
                return -3
            elif count == 1:
                return 0.5
            else:
                return 0.25
        Z = np.zeros([3 for _ in range(d)])
        L = np.zeros([3 for _ in range(d)])
        for a in itertools.product(*[range(3) for _ in range(d)]):
            L[a] = index_to_value(a)
        return np.array([[L, Z], [Z, L]]).transpose(
            (list(range(2, 2 + d)) + [0, 1]))

    # Approximate Laplacian by discrete convolution with a suitably chosen
    # filter matrix.
    return tf.nn.convolution(x,
                             laplacian_filter(2),
                             padding='SAME')


tf.disable_v2_behavior()
N = 500

# Initial Conditions
u_init = np.zeros([1, N, N, 2])
for i in range(500):
    for j in range(500):
        d = (250-i)**2 + (250-j)**2
        u_init[0, i, j, 1] = 5*2.5**(-0.1*d)

# Parameters:
# eps -- time resolution
# U -- the state function
eps = tf.placeholder(tf.float64, shape=())
psi = tf.Variable(u_init, dtype=tf.float64)
img = tf.constant([0, -0.25], dtype=tf.float64)

psi_ = psi + eps * complex_scale(img, laplace(psi))

# Operation to update the state
step = tf.group(psi.assign(psi_))

with tf.Session() as sess:

    # Initialize state to initial conditions
    sess.run(tf.global_variables_initializer())

    for i in range(10**9):
        # Step simulation
        step.run({eps: 0.00015})
        # Visualize every 50 steps
        if i % 5000 == 0:
            clear_output()
            render(psi.eval(), "images/%s.png" %(i // 5000), brightness=0.5)
