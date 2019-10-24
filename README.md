## Tensorflow 2.0 implementation of RNN + attention-based automatic punctuation

Derived from [this project](https://github.com/ottokart/punctuator2) written in Theano.

At this stage this is a rough draft, tested only with a single type of "punctuation"
(actually, sentence boundaries). However, it is easily adapted to any number of punctuation markers.

Also added a little script to infer and visualize the attention weights for any sentence
fed to a trained model.

Hyperparameters and paths to data and checkpoints are written in a bash file to be sourced
before running the script, e.g.

```
source env.sh
python 01_train.python

```

See example_data to see how the data should be formatted.

Preprocessing scripts are not provided in this version.
