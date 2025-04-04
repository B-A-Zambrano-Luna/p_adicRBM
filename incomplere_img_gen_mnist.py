    
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
from keras.datasets import mnist
  
# imgs = imread_collection('./data/figure_incomplete/*.jpg')
# print("Imported", len(imgs), "images")
# imgs = [resize(x,(64,64), mode='constant', anti_aliasing=False) for x in imgs]
# imgs = [1.0*(x>=0.5) for x in imgs]

(train_X, train_y), (test_X, test_y) = mnist.load_data()  #train_x is an array of shape (60000,28,28)
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

imgs = train_X[:16]
n_imgs = len(imgs)
mask_size = 10

plt.figure(figsize=(6, 6))
for i, comp in enumerate(imgs):
    plt.subplot(4, 4, i + 1)
    plt.imshow(comp, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Original Images', fontsize=16)
plt.subplots_adjust(wspace=0.1,hspace=0.1) # width and hight reserved for blank space
plt.show()

x = [random.randint(0, 28 - mask_size -1) for p in range(0, n_imgs)]
y = [random.randint(0, 28 - mask_size -1) for p in range(0, n_imgs)]


plt.figure(figsize=(6, 6))
for i in range(n_imgs):
    img = imgs[i]
    img[x[i]:x[i]+mask_size,y[i]:y[i]+mask_size] = 0
    file_path = "./data/figure_incomplete/"+str(i)+".jpg"
    plt.imsave(file_path, img, cmap=plt.cm.gray)

    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Incomplete Images', fontsize=16)
plt.subplots_adjust(wspace=0.1,hspace=0.1) # width and hight reserved for blank space
plt.show()
    # img = Image.fromarray(img)
    # img.save("./figure_incomplete/"+str(i)+".jpg")


#read the incomplete data from the file
plt.figure(figsize=(6, 6))

for i in range(n_imgs):
    file_path = './data/figure_incomplete/'+str(i)+'.jpg'
    img = plt.imread(file_path)
    plt.subplot(4, 4, i + 1)
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
plt.suptitle('Incomplete Images read from files', fontsize=16)
plt.subplots_adjust(wspace=0.1,hspace=0.1) # width and hight reserved for blank space
plt.show()


# im = asarray(im)
# imfile = io.BytesIO()
# if im.ndim == 3:
#     # if 3D, show as RGB
#     imsave(imfile, im, format="png")
# else:
#     # if 2D, show as grayscale
#     imsave(imfile, im, format="png", cmap=cm.gray)