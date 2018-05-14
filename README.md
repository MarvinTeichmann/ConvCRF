ConvCRF
========
This repository contains the reference implementation for our proposed Convolutional CRFs in PyTorch (Tensorflow planned). The two main entry-points are [demo.py](demo.py) and [benchmark.py](benchmark.py). Demo.py performs ConvCRF inference on a single input image while Benchmark.py compares ConvCRF with FullCRF. Both scripts plot their, yielding an output as shown below.

![Example Output](data/output/Res2.png)

Requirements
-------------

**Plattform**: *Linux, python3 >= 3.4 (or python2 >= 2.7), pytorch 0.4 (or pytorch 0.3 + pyinn), cuda, cudnn*

**Python Packages**: *numpy, imageio, cython, scikit-image, matplotlib*

To install those python packages run `pip install -r requirements.txt` or `pip install numpy imageio cython scikit-image matplotlib`. I recommand using a [python virtualenv][1].




Optional: pyinn, pydensecrf






[1]: https://virtualenvwrapper.readthedocs.io/en/latest/
