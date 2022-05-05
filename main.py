from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

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

    top_bottom_coords = [[y, x] for x in range(corner[1], corner[1]+side) for y in [corner[0], corner[0] + side]]
    side_coords = [[y, x] for y in range(corner[0], corner[0] + side) for x in [corner[1], corner[1] + side]]
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


if __name__ == '__main__':
    files = os.listdir('./image_girl/')
    files.sort()
    init_img = read_img(os.path.join('./image_girl/', files[0]))

    # 1) define initial box
    corner = [20, 50]
    side = 45

    # 2) plot the initial box if wanted
    #f = plt.figure()
    #draw_box(init_img, corner, side)
    #plt.suptitle('orig')

    # 3) find optimal overlay of box in the next image
    img = init_img
    corner0 = corner.copy()
    for _file in files[:]:
        #print('newfile: ',_file)
        new_img = read_img(os.path.join('./image_girl/', _file))
        ref_box = init_img[corner0[0]:corner0[0]+side, corner0[1]:corner0[1]+side]

        winsize = 10
        win_x_min, win_y_min = corner[1]-winsize, corner[0]-winsize
        win_x_max, win_y_max = min(corner[1]+side+winsize, img.shape[1]), min(corner[0]+side+winsize, img.shape[0])
        view_window = new_img[win_y_min:win_y_max, win_x_min:win_x_max]

        # make sure the view window is the right shape
        if view_window.shape[0] < side + 2*winsize:
            print("Type 1 ",_file)
            zeros = np.zeros(shape=(side+2*winsize - view_window.shape[0], view_window.shape[1], 3))
            view_window = np.concatenate([view_window, zeros], axis=0)
        if view_window.shape[1] < side + 2 * winsize:
            print("Type 2 ",_file)
            zeros = np.zeros(shape=(view_window.shape[0], side+2*winsize - view_window.shape[1], 3))
            view_window = np.concatenate([view_window, zeros], axis=1)


        sub_corner = find_optimal_overlay(view_window, ref_box)
        draw_box(view_window, sub_corner, side)
        print(corner)
        corner = (sub_corner[0]+win_y_min, sub_corner[1]+win_x_min)
        print(sub_corner)
        print(corner, sub_corner, win_x_min, win_y_min, img.shape)
        #f = plt.figure()
        #plt.imshow(view_window)
        #plt.suptitle('sub window')
        #f = plt.figure()
        draw_box(new_img, corner, side, filename='./Results/{}'.format(_file))
        #plt.suptitle('full window')
        #img = new_img
        #plt.show()
