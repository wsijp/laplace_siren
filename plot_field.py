#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
import datetime
import argparse


import re

import socket
from pathlib import PurePath
from googleapiclient import discovery
import requests
import google.auth

from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve, polygon_perimeter, rectangle_perimeter)

from tf_siren import SinusodialRepresentationDense, SIRENModel

from PIL import Image, ImageDraw, ImageFont

from matplotlib import pyplot as plt

#from scipy.signal import savgol_filter

PRECISION = 'float32'

DATASET = 'predictions'
STORAGE = 'bq'
PROJECT = 'insight-186822'

BATCH_SIZE = 32

i_inputs_x = 0
i_inputs_y = 1

i_V = 2

i_dx = 3
i_dy = 4

i_ddx = 5
i_ddy = 6

n_columns_y = i_ddy + 1


def make_X(n=1000, size=2., y_init=999.):

    xx = np.linspace(-size/2, size/2, n, dtype='float32')
    x_mesh, y_mesh =np.meshgrid(xx,xx)
    X = np.concatenate([x_mesh.reshape(-1,1), y_mesh.reshape(-1,1)],axis=1)
    y = y_init * np.ones((len(X), n_columns_y), dtype='float32')

    return X, y


def init_y_image(n=1000, init_val=0.):

    img = init_val * np.ones((n, n), dtype='float32')

    return img


def make_text_image(col, row, text='hi', n=400, fontsize=50, font="Arial.ttf"):

    font = ImageFont.truetype(font, size=fontsize)
    image = Image.new("I", (n, n), "black")
    draw = ImageDraw.Draw(image)
    draw.text((col, row), text, font=font)
    return np.asarray(image)


def make_neoval_image(n=400, sentence = 'NEOVAL', value = 1.):


    row=n/2-32 * (n/400)

    fontsize = int( 60*n/400)
    l_text = int(108 * n/400)

    col = (n - (len(sentence) - 0.5) * fontsize)/ 2

    text_move = int(fontsize)

    L = []
    for i, char in enumerate(sentence):
        img = value*(-1)**i * make_text_image(col+i*text_move, row, text=char, n=n, fontsize=fontsize)
        L.append(img)

    return sum(L)




def image_disk(img, x, y, r, value, method=disk):
    """

    rr, cc means y, x
    """

    if method is disk:
        rr, cc = method((y, x), r, shape=img.shape)
    elif method is circle_perimeter:
        rr, cc = method(y, x, int(r), shape=img.shape)

    img[rr, cc] = value

    return img


def image_ellipse_perimeter(img, r, c, r_radius, c_radius, value):
    """

    rr, cc means y, x
    """

    rr, cc = ellipse_perimeter(r, c, r_radius, c_radius)
    #rr, cc = disk((y, x), r, shape=img.shape)
    img[rr, cc] = value

    return img


def image_bounding_box(img, value=0.):

    nn = img.shape

    rr, cc = rectangle_perimeter(start=np.array([1.,1.]), end=np.array([nn[0]-2,nn[1]-2]))

    img[rr, cc] = value

    return img

def image_two_plates(img, value=0.):

    nn = img.shape

    #rr, cc = rectangle_perimeter(start=np.array([1.,1.]), end=np.array([nn[0]-2,nn[1]-2]))

    rr, cc = line(0, 0, nn[0]-1, 0)
    img[rr, cc] = value

    rr, cc = line(0, nn[1]-1, nn[0]-1, nn[1]-1)
    img[rr, cc] = value

    return img


def dipole(img, r=1e-2, d=20, value=1., method=disk):

    nn = img.shape

    x_mid = int(nn[1]/2)
    y_mid = int(nn[0]/2)

    x_pole1 = int((1-d*r) * x_mid)
    x_pole2 = 2 * x_mid - x_pole1

    print(f'x_mid={x_mid}, y_mid={y_mid}, x_pole1={x_pole1}, x_pole2={x_pole2}, R={int(r*nn[0])}')

    img = image_disk(img, x_pole1, y_mid, int(r*nn[0]), -value, method=method)
    img = image_disk(img, x_pole2, y_mid, int(r*nn[0]), value, method=method)

    return img


def crop(img, x_start=0, x_end=None, y_start=0, y_end=None):
    return img[slice(x_start, x_end), slice(y_start, y_end)]


def run(args):
    """

    0   input x1
    1   input x2
    2   V
    3   d2V/dx1^2
    4   d2V/dx2^2

    """

    df = pd.read_csv('/Users/wsijp/Documents/PROJECTS/automatic_differentiation/tmp.csv')

    size = 1.
    n = int(np.sqrt(df.shape[0]))
    X, y = make_X(n=n, size=size)

    # To have black chars correspond to neg charges: minus sign.
    img = -make_neoval_image(n=n, value = 1e1)

    #img = init_y_image(n, init_val = 0.)
    #img = image_bounding_box(img, value = -0.2)
    #img = dipole(img, r=5e-2, d=3, value=1e2, method=circle_perimeter)
    #img = image_disk(img, int(n/2), int(n/2), int(5e-2*n), value=1e2, method=circle_perimeter)


    #r = 5e-2
    #img = image_ellipse_perimeter(img, r=-5e-2, c=5e-2, r_radius=r, c_radius=2*r, value=1e2)


    #img_bb = init_y_image(n, init_val=999.)
    #img_bb = image_bounding_box(img_bb, value=0.0)
    #img_bb = image_two_plates(img_bb, value=0.0)

    #y[:, i_V] = img_bb.reshape(-1)
    #y[:, i_dx] = img_bb.reshape(-1)
    #y[:, i_dy] = img_bb.reshape(-1)

    y[:, i_ddx] = img.reshape(-1)

    E_x = df['3'].values.reshape(n,n)
    E_y = df['4'].values.reshape(n,n)

    E_x[img!=0] = np.nan
    E_y[img!=0] = np.nan

    #lw = np.linalg.norm(np.concatenate([E_x[..., np.newaxis], E_y[..., np.newaxis]], axis=2), axis=2)
    #lw /= lw.max()

    plt.close()
    plt.streamplot(np.arange(n), np.arange(n), E_x, E_y, color='grey', density=2)

    plt.imshow(img, cmap='Greys', interpolation='gaussian')

    plt.ylim([n-10, 10 ])

    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)

    if args.file == 'show':
        plt.show()
    else:
        file_path =args.file
        print(f'Saving figure to {file_path}')
        plt.savefig(file_path)

    return



def make_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', help="Epochs in training.", type = int, default = None)
    parser.add_argument('-N', '--project_namespace', help = "Big Query project and name space in project.namespace format. If no dot-separated path is passed, the argument will be interpreted as a namespace only, and the default project will be prefixed.", default = 'predictions')

    parser.add_argument('--test', action='store_true', help="Run in test mode.", default = False)

    parser.add_argument('-f', '--file', help = "Filename to save to. To show on screen 'show'.", default = 'show' )

    return parser

if __name__ == "__main__":

    parser = make_arg_parser()
    args = parser.parse_args()
    run( args)
