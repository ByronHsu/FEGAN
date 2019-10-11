from sewar.full_ref import rmse, psnr, ssim
import cv2
import numpy as np


def Hough_score(GT, P):
    def calculate(img, minLineLength = 10):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,100,apertureSize = 3)

        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, rho = 1, theta = np.pi/180, threshold = 100, minLineLength = minLineLength, maxLineGap = maxLineGap)

        if lines is None: return 1e-10

        _list = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            dis = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            _list.append(dis)
        l_mean = sum(_list) / len(_list)
        return l_mean
    
    GT_lmean = calculate(GT)
    P_lmean = calculate(P)
    score = (GT_lmean - P_lmean) / P_lmean
    return score

def enlarge_and_crop(img, ratio, size):
    assert(ratio >= 1.0)
    resize_img = cv2.resize(img, None, fx=ratio, fy=ratio)
    h, w, c = resize_img.shape
    cx, cy = h // 2, w // 2
    cropped = resize_img[cy - size // 2: cy + size // 2, cx - size // 2: cx + size // 2,]
    return cropped

def evaluate(GT, P):

    score = {'rmse': 1e9, 'psnr': 0, 'ssim': 0, 'hough': 1e9}
    for ratio in np.arange(1.0, 1.3, 0.05):
        GT_enlarge = enlarge_and_crop(GT, ratio, 256)
        score['rmse'] = min(score['rmse'], rmse(GT_enlarge, P))
        score['psnr'] = max(score['psnr'], psnr(GT_enlarge, P))
        score['ssim'] = max(score['ssim'], ssim(GT_enlarge, P)[0])
        score['hough'] = min(score['hough'], Hough_score(GT_enlarge, P))
    print(score)
    return score