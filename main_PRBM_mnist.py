from option import args
from TreeGen import Image2V
import numpy as np
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import time

from PRBM import Padic_BernoulliRBM

from keras.datasets import mnist


# args.test_only = True

# train_x is an array of shape (60000,28,28)
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# imgs = train_X[:5000]
imgs = train_X[:100]
print("Originally Imported", len(imgs), "images")

image2v = Image2V(args)
start = time.time()
V = image2v.img_tree_array(imgs)
print('success: convert images to 1D tree_based arrays.')
end0 = time.time()
print("tree_array generation time: %.2fs" % (end0 - start))


# n_iterations = args.n_iterations
n_components = args.n_features
h_components = args.h_features
rad_supp = 2
n_iterations = 1000

#n_components = args.n_features
if n_components != pow(args.p, 2*args.l):
    print('warning: the number of visible/hidden units does not match the p-adic tree structure')
    quit()

rbm = Padic_BernoulliRBM(random_state=0, verbose=True,
                         learning_rate=.01,
                         n_iter=n_iterations,
                         n_components=n_components,
                         h_components=h_components,
                         batch_size=50, args=args,
                         rad_supp=rad_supp)
rbm.fit(V)
end1 = time.time()
print("total training time: %.2fs" % (end1 - end0))
