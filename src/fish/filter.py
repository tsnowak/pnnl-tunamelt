
from typing import Optional, List, Tuple
import numpy as np
import cv2
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import normalize
from scipy.fft import fft, fftn, fftfreq, fftshift

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

def mean_filter(video, thresh):
    mean = np.mean(video, axis=0)
    abs_diff = np.abs(video - np.repeat(mean[np.newaxis, :, :], video.shape[0], axis=0))
    out = (abs_diff > thresh) * video
    return out

# consider performing on sparse video - then cluster
def fourier_filter(video, fps, mag_thresh: Optional[float] = .05, f_range: Optional[Tuple] = (1.5, 3.0)):
    '''
    Generates a binary mask of pixels which change at a certain periodicity within the video

    Args:
        video: np.array (N, H, W, C) of video frames to process
        thresh: threshold in hertz of values to pass through the filter
    '''
    # TODO: cut time in half by only working with positive frequency values

    # take the fft of the video
    f_video = fft(video, axis=0, workers=-1)
    # get the sample frequencies of the video
    # these values are only between 0 and 5 hertz due to the nyquist frequency of 10Hz video
    freq = fftfreq(video.shape[0], d=1/fps)

    # Get the frequencies between threshold (aka indices of frequencies we know the turbine spins at)
    s_freq = fftshift(freq)
    i_range = np.argwhere((s_freq > f_range[0]) * (s_freq < f_range[1])).squeeze()
    s_video = fftshift(f_video.real, axes=0)

    # normalize the frequency domain values for filtering
    div_mat = s_video.max(axis=0) + 1e-6 # max of pixel over time (could also do sum)
    div_mat = np.repeat(div_mat[np.newaxis, :, :], f_video.shape[0], axis=0)
    norm_s_video = np.divide(s_video, div_mat)

    # TODO: generalize 350 value - normalize fourier values? - 350
    bool_video = np.abs(s_video[i_range, ...]) > 350
    #bool_video = np.abs(norm_s_video[i_range, ...]) > mag_thresh
    bool_video = bool_video.astype(np.uint8)
    bool_video = np.sum(bool_video, axis=0) > 0
    bool_video = bool_video.astype(np.uint8)*255

    return bool_video, s_video, norm_s_video, s_freq
