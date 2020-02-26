import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

history = model.fit(x=X_train, y=y_train, epochs=30, validation_data=(X_valid, y_valid))

---
Above worked.

Then with:
>>> import pandas as pd
>>> df = pd.DataFrame(history.history)
>>> df.head()
       loss  accuracy  val_loss  val_accuracy
0  0.722916  0.759982  0.510785        0.8290
1  0.494881  0.828127  0.462166        0.8414
2  0.448449  0.841945  0.440557        0.8488
3  0.421495  0.852873  0.392255        0.8648
4  0.400065  0.859582  0.377787        0.8684
>>> import matplotlib.pyplot as plt
>>> df.plot()

Following df.plot(), the python kernel died with the following message:

---
OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
Abort trap: 6

---
Now removing everything associated with pandas from the notebook and re-running, model.fit() appears to be working.

---
So whatever "OMP" is, I suspect something like "open multi processing", what's the fix? Googling the entire above error led to a forum with this:

https://github.com/dmlc/xgboost/issues/1715
"
I tried this and the error stopped !

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
"

"I had the same error on my Mac with a python program using numpy, keras, and matplotlib. I solved it with 'conda install nomkl'."

"My issues were completely unrelated to xgboost, but I got here via google so that I'd share for the sake of others.

I am using keras and matplotlib installed via conda. Setting KMP_DUPLICATE_LIB_OK just changed my experience from a warning to an exception/crash. Install nomkl didn't change anything for me. Eventually I downgraded my version of matplotlib and that fixed things for me
conda install matplotlib=2.2.3"

---
Building things back up step by step, after model.fit(), I can import pandas, and create a DataFrame. The problem happens when I try to plot the DataFrame. So it does seem related to matplotlib, which I know is used by pandas.

So, I may want to try one of the fixes listed here.

Looking at the github repo for this book, I notice it's runnable on Google Colab, so it may get around this problem somehow. Also, there's extra code between the snippets in the book, including plotting some example images from the fashion MNIST dataset.

---
I am trying "conda install nomkl" since it seems the most conda-specific... I actually wound up doing "conda install -c anaconda nomkl" per Anaconda webpage. It failed the first time but worked on the second as follows:

"""
conda install -c anaconda nomkl
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: failed with repodata from current_repodata.json, will retry with next repodata source.
Collecting package metadata (repodata.json): done
Solving environment: done
"""

... Again the kernel dies when trying to plot!

Looks like I'll be trying the os.environ thing. Maybe first I can restart and see if it makes a difference. [it didn't].

So now with the os.environ change, it looks like it's working. I suppose I also have nomkl now. I wonder what the significance of that is.

While I was browsing the web, I found this which may be interesting to know about: https://discuss.pytorch.org/t/two-conda-installs-with-vastly-different-pytorch-performance/17706

Hallelujah, it has worked!