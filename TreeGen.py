from cmath import sqrt
from math import floor
from tkinter import N
import numpy as np
from skimage.transform import resize


def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    return bin(n).replace("0b", "")


def decimal2Pnary(n, p, l):
    if n < 0:
        print('warning: the number should be non-negative!')
        return 0
    s = []
    # if n < p:
    #     s.append(n)
    #     return s
    if n==0:
        s.append(0)
    else:
        while n != 0:
            s.append(n%p)
            n = n//p
    s_len = len(s)
    s =np.array(s)
    
    if s_len < 2*l:
        s = np.concatenate((s,np.zeros(2*l - s_len)))
        return np.array(s)
    elif s_len == 2*l:
        return s
    else:
        print('n value is too big!')
        quit()

    return s

def tree_index_gen(l,p):
    #retun the p_adic indexes for an image of size [p^l, p^l].
    #Satisfies that A[I[i]] = V[i] where A is the image array and V is the visible state 1D vector.
    G_l = pow(p, 2*l)
    I = np.zeros([G_l,2])
    for i in range(G_l):
        b_i = decimal2Pnary(i, p, l)
        if b_i.shape[0] != 2*l:
            print('length of b_i is not correct')
            quit()
        f_i = 0
        g_i = 0
        for j in range(l):
            f_i += int(b_i[2*j])*pow(p,l-1-j) #row
            g_i += int(b_i[2*j+1])*pow(p,l-1-j) #column
        I[i] = [f_i, g_i]
    return I
    # print(self._I)

def tree_index_gen_classic(l,p):
    #retun the p_adic indexes for an image of size [p^l, p^l].
    #Satisfies that A[I[i]] = V[i] where A is the image array and V is the visible state 1D vector.
    G_l = pow(p, 2*l)
    I = np.zeros([G_l,2])
    N = pow(p,l)
    for i in range(G_l):
        f_i = i//N  
        g_i = i%N
        I[i] = [f_i, g_i]
    return I

class Image2V():
    def __init__(self, args):
        self.p = args.p
        self.l = args.l
        self.G_l = pow(self.p,2*self.l)
        # self._I = tree_index_gen(self.l,self.p)
        self._I = tree_index_gen_classic(self.l,self.p)


    def binaryToDecimal(self,val): 
        return int(val, 2) 
    
    
    def image2D2V(self, image):
        #image: array like size [2^l, 2^l]
        V = np.zeros(self.G_l)
        for i in range(self.G_l):
            m,n = self._I[i]
            V[i] = image[int(m), int(n)]
        return V

    def image1D2V(self, image):
        #image: array like size [2^{2l}]
        V = np.zeros(self.G_l)
        N = pow(self.p, self.l)
        # print("N=",N)
        for i in range(self.G_l):
            m,n = self._I[i]
            V[i] = image[int(m*N+n)]
        return V

    def V2image(self, V):
        #V: 1D rray like size [2^{2l}]
        N = pow(self.p, self.l)
        img = np.zeros([N,N])
        for i in range(self.G_l):
            m,n = self._I[i]
            img[int(m),int(n)] = V[i]
        return img

    def img_tree_array(self,imgs):
        if imgs[0].ndim ==3 and imgs[0].shape[2]==3:
            imgs = [0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2] for img in imgs]
        N = pow(self.p,self.l)
        imgs = [resize(x,(N,N), mode='constant', anti_aliasing=True) for x in imgs]
        imgs = [img/img.max() for img in imgs] 
        # #update the data value to binary type.
        imgs = [1.0*(x>=0.3) for x in imgs]
        "Generate the p-adic tree for images of shape [N_imgs,M,N]"
        imgsarr = [x.flatten('C') for x in imgs] #‘C’ means to flatten in row-major (C-style) order.
        imgsarr = [self.image1D2V(x) for x in imgsarr]
        return imgsarr

                
        



        


