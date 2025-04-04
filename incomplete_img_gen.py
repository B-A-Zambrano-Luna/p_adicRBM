    
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import random
from PIL import Image
import os
  
imgs = imread_collection('./data/figure_incomplete/*.jpg')
print("Imported", len(imgs), "images")
imgs = [resize(x,(64,64), mode='constant', anti_aliasing=False) for x in imgs]
imgs = [1.0*(x>=0.5) for x in imgs]

n_imgs = len(imgs)
mask_size = 20

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()
ax[0].imshow(imgs[3])

x = [random.randint(0, 64 - mask_size -1) for p in range(0, n_imgs)]
y = [random.randint(0, 64 - mask_size -1) for p in range(0, n_imgs)]

for i in range(len(imgs)):
    img = imgs[i]
    img[x[i]:x[i]+mask_size,y[i]:y[i]+mask_size] = 0
    file_path = "./data/figure_incomplete/"+str(i)+".jpg"
    plt.imsave(file_path, img, cmap=plt.cm.gray)
    # img = Image.fromarray(img)
    # img.save("./figure_incomplete/"+str(i)+".jpg")

ax[1].imshow(imgs[3])

plt.show()

image_file = './data/figure_incomplete/0.jpg'
image = plt.imread(image_file)


# im = asarray(im)
# imfile = io.BytesIO()
# if im.ndim == 3:
#     # if 3D, show as RGB
#     imsave(imfile, im, format="png")
# else:
#     # if 2D, show as grayscale
#     imsave(imfile, im, format="png", cmap=cm.gray)