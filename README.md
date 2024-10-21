# Han-sur-Lesse Winterschool Lecture on Plane Wave Density Functional Theory

[![build](https://github.com/ifilot/hsl-pwdft-exercises/actions/workflows/build.yml/badge.svg)](https://github.com/ifilot/hsl-pwdft-exercises/actions/workflows/build.yml)

This repository contains the set of practice exercises for the [Han-sur-Lesse
Winterschool](https://www.han-sur-lesse-winterschool.nl/) lecture on plane-wave
density functional theory. For the problem formulation and the lecture notes,
head over to the downloads section below and download the required documents.

## Downloads

* [Exercises PDF](https://github.com/ifilot/hsl-pwdft-exercises-latex/releases/latest/download/hsl-2024-pwdft-exercises.pdf)
* [Lecture notes on plane-wave density functional theory](https://github.com/ifilot/pwdft-lecture-notes/releases/latest/download/pwdft-filot.pdf)

## Dependencies

These exercises require relatively few modules. A list is provided below:
* NumPy
* SciPy
* pyFFTW
* [PyPWDFT](https://pypwdft.imc-tue.nl/).

### Anaconda installation

```bash
conda install numpy scipy pyfftw
conda install -c ifilot pypwdft
```

### PyPI installation

```bash
pip install numpy scipy pyfftw pypwdft
```