# -*- coding: utf-8 -*-
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

# This renders the images.
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

# Complex multiplication.
def complex_mul(z1, z2):
    """Multiply two complex tensors."""
    x1, y1 = tf.split(z1, [1, 1], -1)
    x2, y2 = tf.split(z2, [1, 1], -1)
    return tf.concat((x1 * x2 - y1 * y2,
                      x1 * y2 + x2 * y1), -1)

# Discrete Laplacian.
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
                return -4.0
            elif count == 1:
                return 0.5
            else:
                return 0.5
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
N = 200

# Generate the inital state.
psi_init = np.zeros([1, N, N, 2])
x_0, y_0 = (30, 0)
for i in range(N):
    for j in range(N):
        d = ((N-y_0)//2-i)**2 + ((N-x_0)//2-j)**2
        v = 10**(-5*10e-3*d)
        thr = 10e-2
        psi_init[0, i, j, 0] = v

# Generate the potential that confines the particle to a box.
v_init = np.zeros([1, N, N, 2])
for i in range(N):
    for j in range(N):
        x, y = N // 2 - i , N // 2 - j
        v = -10e-0 / np.sqrt(max(x**2 + y**2, 0.1))
        v_init[0, i, j, 0] = v

# Define some tensorflow values.
eps = tf.placeholder(tf.float64, shape=())
psi = tf.Variable(psi_init, dtype=tf.float64)
v = tf.constant(v_init, dtype=tf.float64)
c1 = tf.constant([0, -0.25], dtype=tf.float64)
c2 = tf.constant([0, 0.5], dtype=tf.float64)

# Some simulation parameters
plot_step = 10e2
resolution = 10e-4
brightness = 0.01

# The update rule derived from the Schrödinger equation
psi_ = psi + eps * (complex_mul(c1, laplace(psi)) + complex_mul(c2, complex_mul(v, psi)))
step = tf.group(psi.assign(psi_))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Run the simulation
    for i in range(10**9):
        step.run({eps: resolution})
        # Produce a plot every plot_step steps.
        if i % plot_step == 0:
            clear_output()
            render(psi.eval(), "images/%s.png" %(int(i) // int(plot_step)), brightness=brightness)
