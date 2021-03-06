from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def read_img(filename):
    img = Image.open(filename)
    return np.array(img)


def draw_box(img, corner, side, filename=None):
    '''

    plots imshow of img with a box defined by (corner, side)
    :param img: 2D np.array of image
    :param corner: coordinates of corner of box
    :param side: side length of box
    :param filename: if not None, save image to filename
    '''

    top_bottom_coords = [[y, x] for x in range(corner[1], min(img.shape[1], corner[1]+side)) for y in [corner[0], min(img.shape[0], corner[0] + side)]]
    side_coords = [[y, x] for y in range(corner[0], min(img.shape[0], corner[0] + side)) for x in [corner[1], min(img.shape[1], corner[1] + side)]]
    for coords in top_bottom_coords + side_coords:
        try:
            img[coords[0], coords[1]] = 0
        except:
            pass
    if filename is None:
        plt.imshow(img)
    else:
        f = plt.figure()
        plt.imshow(img)
        plt.savefig(filename)
        plt.close(f)
    return


def SSD(a1, a2):
    '''

    :param a1: np array 1
    :param a2: np array 2 (matching dims to a1)
    :return: float. mean square error across all dimensions
    '''
    assert (a1.shape == a2.shape)
    return np.sum(np.power((a1-a2), 2))


def CC(a1, a2):
    '''

    :param a1: np array 1
    :param a2: np array 2 (matching dims to a1)
    :return: float. Cross correlation of arrays a1 and a2
    '''
    assert (a1.shape == a2.shape)
    return -np.sum(a1*a2)


def NCC(a1, a2):
    '''

    :param a1: np array 1
    :param a2: np array 2 (matching dims to a1)
    :return: float. Normalised Cross correlation of arrays a1 and a2
    '''
    assert (a1.shape == a2.shape)
    a1 = a1 - np.mean(a1)
    a2 = a2 - np.mean(a2)
    return -np.sum(a1*a2) / np.sqrt(np.sum(a1**2) * np.sum(a2**2))



def find_optimal_overlay(img, ref, scorefunc = SSD):
    '''

    :param img: np.array image to find optimal position on
    :param ref: np.array reference snipped of image
    :param scorefunc: function. quantify difference between like-dimension arrays as float
    :return: y,x coords of the maximally likely overlay of ref on img
    '''
    best_score = 1e10
    best_coords = (0, 0)
    im = np.zeros(shape=img.shape[:2])
    for i in range(img.shape[1] - ref.shape[1]):
        for j in range(img.shape[0] - ref.shape[0]):
            base = img[j:j+ref.shape[0], i:i+ref.shape[1]]
            score = scorefunc(base, ref)
            im[j, i] = score
            if score < best_score:
                best_score = score
                best_coords = (j, i)
    return best_coords


def jpg_to_mp4(path, outdir, video_name='NCC.mp4'):
    '''

    converts directory of jpgs into a mp4, sorting images by filename
    :param path: path to jpg images
    :param outdir: directory to save video in
    '''

    img_array = []
    files = os.listdir(path)
    files.sort()
    for filename in tqdm(files):
        img = cv2.imread(os.path.join(path, filename))
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(os.path.join(outdir, video_name), cv2.VideoWriter_fourcc(*'MP4V'), 15, size)

    for i in tqdm(range(len(img_array))):
        out.write(img_array[i])
    out.release()
    return


if __name__ == '__main__':

    metric = CC # choose SSD, CC or NCC

    files = os.listdir('./image_girl/')
    files.sort()
    init_img = read_img(os.path.join('./image_girl/', files[0]))

    # 1) define initial box
    corner = [20, 50]
    side = 45

    # 3) find optimal overlay of box in the next image
    img = init_img
    corner0 = corner.copy()
    for _file in files[:]:
        new_img = read_img(os.path.join('./image_girl/', _file))
        ref_box = init_img[corner0[0]:corner0[0]+side, corner0[1]:corner0[1]+side]

        winsize = 10
        win_x_min, win_y_min = corner[1]-winsize, corner[0]-winsize
        win_x_max, win_y_max = min(corner[1]+side+winsize, img.shape[1]), min(corner[0]+side+winsize, img.shape[0])
        view_window = new_img[win_y_min:win_y_max, win_x_min:win_x_max]

        # make sure the view window is the right shape
        if view_window.shape[0] < side + 2*winsize:
            zeros = np.zeros(shape=(side+2*winsize - view_window.shape[0], view_window.shape[1], 3))
            view_window = np.concatenate([view_window, zeros], axis=0)
        if view_window.shape[1] < side + 2 * winsize:
            zeros = np.zeros(shape=(view_window.shape[0], side+2*winsize - view_window.shape[1], 3))
            view_window = np.concatenate([view_window, zeros], axis=1)

        sub_corner = find_optimal_overlay(view_window, ref_box, scorefunc=metric)
        draw_box(view_window, sub_corner, side)
        corner = (sub_corner[0]+win_y_min, sub_corner[1]+win_x_min)
        draw_box(new_img, corner, side, filename='./Results/images/{}'.format(_file)) # this will save images as jpgs in ./Results/images/

    # finally convert to video
    jpg_to_mp4('./Results/images/', './Results/')