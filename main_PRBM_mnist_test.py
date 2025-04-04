import time
import numpy as np
from option import args
from PRBM import Padic_BernoulliRBM
from keras.datasets import mnist
from TreeGen import Image2V
import matplotlib.pyplot as plt
import os
from TreeGen import Image2V
from skimage.io import imread_collection, imshow


n_iterations = args.n_iterations
n_components = args.n_features
h_components = args.h_features
N_chain_gibbs = 1
epoch = 1000
image2v = Image2V(args)


# train_x is an array of shape (60000,28,28)
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

plt.figure(figsize=(6, 6))
for i, comp in enumerate(train_X[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Original Images', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

imgs = train_X[:100]
print("Originally Imported", len(imgs), "images")
image2v = Image2V(args)
start = time.time()
V = image2v.img_tree_array(imgs)
print('success: convert images to 1D tree_based arrays.')
end0 = time.time()
print("tree_array generation time: %.2fs" % (end0 - start))

rbm = Padic_BernoulliRBM(random_state=0,
                         verbose=True,
                         learning_rate=.01,
                         n_iter=n_iterations,
                         n_components=n_components,
                         h_components=h_components,
                         batch_size=50,
                         args=args)


apath = './saved_models/'
# epoch = 3000
save_dirs = [apath]
for s in save_dirs:  # torch.save(model.state_dict(), PATH)
    rbm.components_ = np.loadtxt(os.path.join(
        s, 'model_epoch{}_w.csv'.format(epoch)), delimiter=',')
    rbm.intercept_visible_ = np.loadtxt(os.path.join(
        s, 'model_epoch{}_a_visible.csv'.format(epoch)), delimiter=',')
    rbm.intercept_hidden_ = np.loadtxt(os.path.join(
        s, 'model_epoch{}_b_hidden.csv'.format(epoch)), delimiter=',')


plt.figure(figsize=(8, 8))

plt.subplot(1, 2, 1)
plt.imshow(rbm.components_.reshape((27, 27)),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.xticks(())
plt.yticks(())
# plt.suptitle('N components extracted by RBM', fontsize=16)
# plt.subplots_adjust(wspace=0.1,hspace=0.1) # width and hight reserved for blank space

plt.subplot(1, 2, 2)
plt.imshow(image2v.V2image(rbm.components_),
           cmap=plt.cm.gray_r, interpolation='nearest')
plt.xticks(())
plt.yticks(())
plt.suptitle('N components extracted by RBM', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()

# x_latent=rbm.transform(x)
# x_latent.shape

gen_x = rbm.gibbs_sampling(1, np.array(V[:100]))

# gen_x = np.array(V[:100])
gen_x = [image2v.V2image(x) for x in gen_x]

plt.figure(figsize=(6, 6))
for i, comp in enumerate(gen_x):
    # for i in range(4):
    plt.subplot(10, 10, i + 1)
    # plt.imshow(gen_x[i].reshape((27, 27)), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.imshow(gen_x[i], cmap=plt.cm.gray_r, interpolation='nearest')

    plt.xticks(())
    plt.yticks(())
plt.suptitle('Reconstructed input images by Gibbs sampling', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

z = np.random.random_sample((2, 729))
# plot random noise
plt.figure(figsize=(6, 6))
for i, comp in enumerate(z):
    plt.subplot(1, 2, i + 1)
    plt.imshow(comp.reshape((27, 27)), cmap=plt.cm.gray_r,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Random Noise', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)

plt.show()

n_imgs = 16
for i in range(n_imgs):
    file_path = './data/figure_incomplete/'+str(i)+'.jpg'
    img = plt.imread(file_path)
    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Incomplete Images read from files', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()

imgs_incomplete = imread_collection('./data/figure_incomplete/*.jpg')
# image2v = Image2V(args)
start = time.time()
V = image2v.img_tree_array(imgs_incomplete)
V = np.array(V)
imgs_complete = rbm.gibbs_sampling(N_chain_gibbs, V)
imgs_complete = [image2v.V2image(x) for x in imgs_complete]

for i in range(n_imgs):
    img = imgs_complete[i]
    file_path = './data/figure_completed/'+str(i)+'.jpg'
    plt.imsave(file_path, img, cmap=plt.cm.gray)

    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Completed Images', fontsize=16)
# width and hight reserved for blank space
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.show()
