
import cv2

def template_match( img: np.array,
                    template: np.array,
                    method='cv.TM_SQDIFF_NORMED'):

    res = cv.matchTemplate(img, template, method)
    _, _, top_left, _ = cv.minMaxLoc(res)

    w, h = template.shape[::-1]


    return  