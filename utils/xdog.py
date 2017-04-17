import cv2
import numpy as np
import argparse

#Extract line graph by XDoG algorithm

def dog(img, sigma, k_sigma, p):
    sigma_large = sigma * k_sigma
    G_small = cv2.GaussianBlur(img,(0, 0), sigma)
    G_large = cv2.GaussianBlur(img,(0, 0), sigma_large)
    S = G_small - p * G_large
    return S

def soft_threshold(SI, epsilon, phi):
    T = np.zeros(SI.shape)
    SI_bright = SI >= epsilon
    SI_dark = SI < epsilon
    T[SI_dark] = 1.0
    T[SI_bright] = 1.0 + np.tanh( phi * (SI[SI_bright]))
    return T

def _XDoG(img, sigma, k_sigma, p, epsilon, phi):
    S = dog(img, sigma, k_sigma, p)
    T = soft_threshold(S, epsilon, phi)
    return (T*127.5).astype(np.uint8)

def XDoG(img, sigma=0.5, k_sigma=1.6, p=0.98, epsilon=-0.1, phi=200):
    if len(img.shape) == 3 and img.shape[2]==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    img=img.astype(np.float32)/255.0
    return _XDoG(gray,sigma=sigma, k_sigma=k_sigma, p=p, epsilon=epsilon, phi=phi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Exetract Line Image')
    parser.add_argument('--input', '-i', default='', help='input image.')
    parser.add_argument('--output', '-o', default='result.jpg', help='output image')
    args = parser.parse_args()
    img = cv2.imread(args.input)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = XDoG(gray)
    print(cv2.imwrite(args.output, result))
