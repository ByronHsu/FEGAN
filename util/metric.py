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

def evaluate(GT, P):
    score_rmse = rmse(GT, P)
    score_psnr = psnr(GT, P)
    score_ssim = ssim(GT, P)[0]
    score_hough = Hough_score(GT, P)

    return {
        'rmse': score_rmse,
        'psnr': score_psnr,
        'ssim': score_ssim,
        'hough': score_hough
    }