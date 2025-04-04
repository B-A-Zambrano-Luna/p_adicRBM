import time
import numpy as np
from option import args
from PRBM import Padic_BernoulliRBM
from keras.datasets import mnist
import matplotlib.pyplot as plt
import os
from TreeGen import Image2V
from numpy import linalg as LA
from skimage.transform import resize

n_iterations = args.n_iterations
n_components = args.n_features
image2v = Image2V(args)


# train_x is an array of shape (60000,28,28)
(train_X, train_y), (test_X, test_y) = mnist.load_data()
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# plt.figure(figsize=(6, 6))
# for i, comp in enumerate(train_X[:100]):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape((27, 27)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('Original Images', fontsize=16)
# # width and hight reserved for blank space
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()

imgs = train_X[:100]
imgs = [resize(x, (27, 27), mode='constant', anti_aliasing=True) for x in imgs]
imgs = np.array(imgs)
print("Originally Imported", len(imgs), "images")
image2v = Image2V(args)
start = time.time()
V = image2v.img_tree_array(imgs)
print('success: convert images to 1D tree_based arrays.')
end0 = time.time()
print("tree_array generation time: %.2fs" % (end0 - start))


apath = './saved_models/'
# epoch = 3000
epoch = 1000
save_dirs = [apath]
results = []
for k in range(1, 7):
    rbm = Padic_BernoulliRBM(random_state=0,
                             verbose=True,
                             learning_rate=.01,
                             n_iter=n_iterations,
                             n_components=n_components,
                             batch_size=50,
                             rad_supp=k,
                             args=args)

    for s in save_dirs:  # torch.save(model.state_dict(), PATH)
        rbm.components_ = np.loadtxt(os.path.join(
            s, 'model_epoch{}_w_k_{}.csv'.format(epoch, k)), delimiter=',')
        rbm.intercept_visible_ = np.loadtxt(os.path.join(
            s, 'model_epoch{}_a_visible_k_{}.csv'.format(epoch, k)), delimiter=',')
        rbm.intercept_hidden_ = np.loadtxt(os.path.join(
            s, 'model_epoch{}_b_hidden_k_{}.csv'.format(epoch, k)), delimiter=',')

    gen_x = rbm.gibbs_sampling(1, np.array(V[:100]))

    # gen_x = np.array(V[:100])
    gen_x = [image2v.V2image(x) for x in gen_x]
    errors = [LA.norm(gen_x[i] - imgs[i])
              for i in range(imgs.shape[0])]
    errors = np.array(errors)
    results.append(errors.sum()/imgs.shape[0])

levels = [k for k in range(1, 7)]

# plotting the points
plt.plot(levels, results, color='black', linestyle='dashed', linewidth=3,
         marker='o', markerfacecolor='blue', markersize=12)

# setting x and y axis range
# plt.ylim(0, 1)
# plt.xlim(0, 7)

# naming the x axis
plt.ylabel('Errors')
# naming the y axis
plt.xlabel('$\log_3$ of Number of parameters')

# giving a title to my graph
plt.title('Erros p-adic RBM')

# function to show the plot
plt.show()
