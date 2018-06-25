ConvCRF
========
This repository contains the reference implementation for our proposed [Convolutional CRFs][4] in PyTorch (Tensorflow planned). The two main entry-points are [demo.py](demo.py) and [benchmark.py](benchmark.py). Demo.py performs ConvCRF inference on a single input image while benchmark.py compares ConvCRF with FullCRF. Both scripts output plots similar to the one shown below.

![Example Output](data/output/Res2.png)

Requirements
-------------

**Plattform**: *Linux, python3 >= 3.4 (or python2 >= 2.7), pytorch 0.4 (or pytorch 0.3 + pyinn), cuda, cudnn*

**Python Packages**: *numpy, imageio, cython, scikit-image, matplotlib*

To install those python packages run `pip install -r requirements.txt` or `pip install numpy imageio cython scikit-image matplotlib`. I recommand using a [python virtualenv][1].

### Optional Packages: pyinn, pydensecrf

[**Pydensecrf**][2] is required to run FullCRF, which is only needed for the benchmark. To install pydensecrf, follow the instructions [here][2] or simply run `pip install git+https://github.com/lucasb-eyer/pydensecrf.git`. **Warning** Running `pip install git+` downloads and installs external code from the internet.

[**PyINN**][3] allows us to write native cuda operations and compile them on-the-fly during runtime. PyINN is used for our initial ConvCRF implementation and required for PyTorch 0.3 users. PyTorch 0.4 introduces an Im2Col layer, making it possible to implement ConvCRFs entirely in PyTorch. PyINN can be used as alternative backend. Run `pip install git+https://github.com/szagoruyko/pyinn.git@master` to install PyINN.


Execute
--------

**Demo**: Run `python demo.py data/2007_001288_0img.png data/2007_001288_5labels.png` to perform ConvCRF inference on a single image. Try `python demo.py --help` to see more options.

**Benchmark**: Run `python benchmark.py data/2007_001288_0img.png data/2007_001288_5labels.png` to compare the performance of ConvCRFs to FullCRFs. This script will also tell you how much faster ConvCRFs are. On my system ConvCRF7 is more then **40** and ConvCRF5 more then **60** times faster.


Citation
--------
If you benefit from this project, please consider citing our [paper][4]. 

TODO
-----

- [x] Build a native PyTorch 0.4 implementation of ConvCRF
- [x] Provide python 2 implementation
- [ ] Build a Tensorflow implementation of ConvCRF



[1]: https://virtualenvwrapper.readthedocs.io/en/latest/
[2]: https://github.com/lucasb-eyer/pydensecrf
[3]: https://github.com/szagoruyko/pyinn
[4]: https://arxiv.org/abs/1805.04777


where:
*   NAME: an arbitrary runname
*   DELF_OAG: path to object_annotation_group created by the delf pipeline
*   DET_EDG: path to eval_detection_group created by the detection pipeline
