ConvCRF
========
This repository contains the reference implementation for our proposed Convolutional CRFs. The implementation uses the PyTorch framework (a Tensorflow implementation is also planned). The two main entry-points are [demo.py](demo.py) and [benchmark.py](benchmark.py). Demo.py performs ConvCRF inference on a single input image. Benchmark.py compares the output of ConvCRF to FullCRFs. Both scripts plot their, yielding an output as shown below. Speed-comparison can be produced using the '--speed_test' flag.


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