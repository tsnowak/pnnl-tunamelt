from skimage.restoration import denoise_wavelet
import numpy as np
import cv2

# not very effective on these low-res images
def wavelet_denoising(x, method='BayesShrink'):
    
    x = denoise_wavelet(x, multichannel=True, convert2ycbcr=False,
        method=method, mode='soft', rescale_sigma=True)

    return x

# crop polygon of image, and return with black background
def crop_polygon(img, pts):
    ## (1) Crop the bounding rect
    rect = cv2.boundingRect(pts)
    x,y,w,h = rect
    croped = img[y:y+h, x:x+w].copy()

    ## (2) make mask
    pts = pts - pts.min(axis=0)

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    ## (3) do bit-op
    dst = cv2.bitwise_and(croped, croped, mask=mask)

    return dst

class DiffSub():

    def __init__(self, n_frames = 2, diff_thresh=5):
        self.n_frames = n_frames
        self.diff = None
        self.diff_thresh
        self.buff = []

    def update(self, x):
        if self.diff is None:
            self.buff.append(x)
            return x

    def diff_current(self, x):
        diff = sum([x - f for f in self.buff])
        return diff

    def diff_neighbor(self,):
        pass
