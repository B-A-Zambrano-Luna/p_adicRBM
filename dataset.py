
from skimage.io import imread_collection, imshow
from skimage.transform import resize
import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d

def data_augmentation(image):
    # out = np.transpose(image, (1,2,0))
    out = image
    mode = np.random.randint(0,8)
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return out


def get_horse_loaders(imgs,args):
    # imgs = imread_collection('./data/figure/*.jpg')
    len_data = len(imgs)
    train_patches = []
    # scales = [1, 0.9, 0.8, 0.7]
    patches_per_image = args.patch_per_image
    patch_size = args.patch_size

    # imgs = [resize(x,(64,64), mode='constant', anti_aliasing=False) for x in imgs]


    for i in range(len_data):
        image = imgs[i]
        height = image.shape[0]
        width =  image.shape[1]
        # for scale in scales:
            # img = resize(image,(np.floor(height*scale),np.floor(height*scale)), mode='constant', anti_aliasing=False) 
        patches = extract_patches_2d(image, (min(patch_size*2,height),min(patch_size*2,width)), max_patches=patches_per_image) #patch.shape = (16, 64, 64)
        patches = [resize(x,(64,64), mode='constant', anti_aliasing=False) for x in patches]
        patches = [data_augmentation(x) for x in patches]
        train_patches.append(patches)
    train_patches = np.array(train_patches) #shape = (len_data, 16, 64, 64)
    train_patches = train_patches.reshape((-1,) + train_patches.shape[-2:]) #shape = (160*len_data, 64, 64)
    a = 1
    return train_patches
# get_horse_loaders()


