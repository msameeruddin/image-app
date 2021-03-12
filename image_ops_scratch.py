import cv2
import numpy as np
import json
import base64

from matplotlib import pyplot as plt


def read_image_string(contents):
   encoded_data = contents[0].split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   return img


class ImageOperations(object):
    def __init__(self, image_file_src):
        self.image_file_src = image_file_src
        self.MAX_PIXEL = 255
        self.MIN_PIXEL = 0
        self.MID_PIXEL = self.MAX_PIXEL // 2
    
    def read_this(self, gray_scale=False):
        image_src = self.image_file_src
        if gray_scale:
            image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
        return image_src
    
    def mirror_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_mirror = np.fliplr(image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_mirror, head_text='Mirrored', gray_scale=gray_scale)
            return None
        return image_mirror
    
    def flip_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_flip = np.flipud(image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_flip, head_text='Flipped', gray_scale=gray_scale)
            return None
        return image_flip
    
    def equalize_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        if not gray_scale:
            r_image = image_src[:, :, 0]
            g_image = image_src[:, :, 1]
            b_image = image_src[:, :, 2]

            r_image_eq = cv2.equalizeHist(r_image)
            g_image_eq = cv2.equalizeHist(g_image)
            b_image_eq = cv2.equalizeHist(b_image)

            image_eq = np.dstack(tup=(r_image_eq, g_image_eq, b_image_eq))
        else:
            image_eq = cv2.equalizeHist(image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_eq, head_text='Equalized', gray_scale=gray_scale)
            return None
        return image_eq
    
    def convert_binary(self, image_matrix, thresh_val):
        color_1 = self.MAX_PIXEL
        color_2 = self.MIN_PIXEL
        initial_conv = np.where((image_matrix <= thresh_val), image_matrix, color_1)
        final_conv = np.where((initial_conv > thresh_val), initial_conv, color_2)
        return final_conv

    def binarize_this(self, with_plot=False, gray_scale=False, colors=None):
        image_src = self.read_this(gray_scale=gray_scale)
        image_b = self.convert_binary(image_matrix=image_src, thresh_val=self.MID_PIXEL)

        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_b, head_text='Binarized', gray_scale=gray_scale)
            return None
        return image_b
    
    def invert_this(self, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        image_i = ~ image_src
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_i, head_text='Inverted', gray_scale=gray_scale)
            return None
        return image_i

    def solarize_this(self, thresh_val=128, with_plot=False, gray_scale=False):
        image_src = self.read_this(gray_scale=gray_scale)
        if not gray_scale:
            r_image, g_image, b_image = image_src[:, :, 0], image_src[:, :, 1], image_src[:, :, 2]
            r_sol = np.where((r_image < thresh_val), r_image, ~r_image)
            g_sol = np.where((g_image < thresh_val), g_image, ~g_image)
            b_sol = np.where((b_image < thresh_val), b_image, ~b_image)
            image_sol = np.dstack(tup=(r_sol, g_sol, b_sol))
        else:
            image_sol = np.where((image_src < thresh_val), image_src, ~image_src)
        
        if with_plot:
            self.plot_it(orig_matrix=image_src, trans_matrix=image_src, head_text='Solarized', gray_scale=gray_scale)
            return None
        return image_sol
    
    def plot_it(self, orig_matrix, trans_matrix, head_text, gray_scale=False):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 20))
        cmap_val = None if not gray_scale else 'gray'
        
        ax1.axis("off")
        ax1.title.set_text('Original')
        
        ax2.axis("off")
        ax2.title.set_text(head_text)
        
        ax1.imshow(orig_matrix, cmap=cmap_val)
        ax2.imshow(trans_matrix, cmap=cmap_val)
        plt.show()
        return True


if __name__ == '__main__':
    image = cv2.imread('lena_original.png', 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imo = ImageOperations(image_file_src=image)
    imo.binarize_this(with_plot=True, gray_scale=False)