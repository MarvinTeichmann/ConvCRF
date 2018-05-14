ConvCRF
========
This repository contains the reference implementation for our proposed Convolutional CRFs in PyTorch (Tensorflow planned). The two main entry-points are [demo.py](demo.py) and [benchmark.py](benchmark.py). Demo.py performs ConvCRF inference on a single input image while Benchmark.py compares ConvCRF with FullCRF. Both scripts plot their, yielding an output as shown below.

![Example Output](data/output/Res2.png)

# Too Early (Bird)

Wow, you have found this repository very early. The code is fully functional, but we are still working on the documentation.

Requirements


* python3,
* pytorch 0.3
* cython
* pydensecrf

* scikit-image
* matplotlib
