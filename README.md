# Schrödinger equation simulation

[Particle in a box](https://en.wikipedia.org/wiki/Particle_in_a_box) is a famous ideal model in quantum mechanics. Here we simulate what a particle in a box looks like.

The time evolution of a wave function is described by the [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation). This script simulates its solution by a finite element method. A wave function is complex-valued, the colors are produced by converting a complex number to an RGB value (see https://en.wikipedia.org/wiki/Domain_coloring#Method).

This script is loosely based on a standard [TensorFlow PDE example](https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/tutorials/pdes/index.md).

## Running

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
main.py
```

## Converting the rendered frames to a video

After running the simulation:
```
cd images
for ((i=0; i<=0; i++)) ; do mv $i.png `printf %04d.png $i` ; done  # Zero padded filenames for ffmpeg.
ffmpeg -r 60 -f image2 -s 500x500 -i %04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
```
Notice that here we use 4 decimal filenames (e.g. "0020.png"). If you have over 10 000 frames replace `%04d` with `%05d` above.
