ConvCRF
========
This repository contains the reference implementation for our proposed Convolutional CRFs in PyTorch (Tensorflow planned). The two main entry-points are [demo.py](demo.py) and [benchmark.py](benchmark.py). Demo.py performs ConvCRF inference on a single input image while benchmark.py compares ConvCRF with FullCRF. Both scripts output plots as shown below.

![Example Output](data/output/Res2.png)

Requirements
-------------

**Plattform**: *Linux, python3 >= 3.4 (or python2 >= 2.7), pytorch 0.4 (or pytorch 0.3 + pyinn), cuda, cudnn*

**Python Packages**: *numpy, imageio, cython, scikit-image, matplotlib*

To install those python packages run `pip install -r requirements.txt` or `pip install numpy imageio cython scikit-image matplotlib`. I recommand using a [python virtualenv][1].

### Optional Packages: pyinn, pydensecrf

[**Pydensecrf**][2] is required in order to run FullCRF, which is only needed for the benchmark. To install pydensecrf, follow the instructions [here][2] or simply run `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`. **Warning** Running `pip install git+` downloads and installs external code from the internet.

[**PyINN**][3] integrates CuPy into PyTorch. This allows us to write native cuda operations and compile them on-the-fly during runtime. PyINN is the basis of your initial ConvCRF implementation. PyTorch 0.4 introduces an Im2Col layer, making it possible to implement ConvCRFs entirely in PyTorch. PyINN can be used as alternative backend. Run `pip install git+https://github.com/szagoruyko/pyinn.git@master` to install PyINN.


Execute
--------

**Inference**: Run `python inference.py data/2007_001288_0img.png data/2007_001288_5labels.png` to perform ConvCRF inference on a single image. Try `python inference.py --help` to see more options.

**Benchmark**: Run `python benchmark.py data/2007_001288_0img.png data/2007_001288_5labels.png` to compare the performance of ConvCRFs to FullCRFs. This script will also tell you how much faster ConvCRFs are. On my system I get a speed-up factor of **31**.

TODO
-----

- [x] Build a native PyTorch 0.4 implementation of ConvCRF
- [ ] Add more comments to ConvCRF class
- [ ] Add setup.py script and reorganize repository
- [ ] Build a Tensorflow Implementation of ConvCRF



[1]: https://virtualenvwrapper.readthedocs.io/en/latest/
[2]: https://github.com/lucasb-eyer/pydensecrf
[3]: https://github.com/szagoruyko/pyinn
