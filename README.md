# Schrödinger equation simulation

[Particle in a box](https://en.wikipedia.org/wiki/Particle_in_a_box) is a famous ideal model in quantum mechanics. Here we simulate what a particle in a box looks like.

The time evolution of a wave function is described by the [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation). This script simulates its solution by a finite element method. A wave function is complex-valued, the colors are produced by converting a complex number to an RGB value (see https://en.wikipedia.org/wiki/Domain_coloring#Method).

This script is loosely based on a standard [TensorFlow PDE example](https://en.wikipedia.org/wiki/Particle_in_a_box).

# Running

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
main.py
```
