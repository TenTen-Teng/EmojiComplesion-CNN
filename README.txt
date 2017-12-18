EmojiComplesion-CNN
Use CNN model to implement emoji classification

Acknowledgement:
Yassine Benajiba (George Washington University, Intro to Statistical NLP, CSCI 6907)

Requirements
Python 3
Tensorflow > 0.12
Numpy

Training
Train:
./train.py

Evaluating
```bash
./eval.py --eval_train --checkpoint_dir="./runs/1513545947/checkpoints/"
Replace the checkpoint dir with the output from the training. To use your own data, change the eval.py script to load your data.

References
https://github.com/dennybritz/cnn-text-classification-tf
http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/