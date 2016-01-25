Character-Aware Neural Language Models in Tensorflow
====================================================

Tensorflow implementation of [Character-Aware Neural Language Models](http://arxiv.org/abs/1508.06615). The original torch code and dataset can be found [here](https://github.com/yoonkim/lstm-char-cnn).

![model.png](./assets/model.png)


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [Tensorflow](https://www.tensorflow.org/)


Usage
-----

To train a model with `ptb` dataset:

    $ python main.py --dataset ptb --is_train True

To test an existing model:

    $ python main.py --dataset ptb

(in progress)


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
