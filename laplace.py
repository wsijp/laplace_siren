#!/usr/bin/env python3
import numpy as np
import pandas as pd
import argparse

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers

import pickle


from skimage.draw import (line, polygon, disk,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve, polygon_perimeter, rectangle_perimeter)

from tf_siren import SinusodialRepresentationDense, SIRENModel

from PIL import Image, ImageDraw, ImageFont


i_inputs_x = 0
i_inputs_y = 1

i_V = 2

i_dx = 3
i_dy = 4

i_ddx = 5
i_ddy = 6

n_columns_y = i_ddy + 1

negloglik = lambda y, rv_y: -rv_y.log_prob(y)


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



def my_loss(y_true, y_pred_inputs):
    """
    State (theta, w) angular position, angular velocity

    x = y_pred_inputs[:, 1]
    y = y_pred_inputs[:, 2]
    z = y_pred_inputs[:, 3]

    interior (away from boundary) points are marked by 999

    """

    #SCALE = 1e-2

    inputs = y_pred_inputs[:, i_inputs_x:i_inputs_y+1]
    V = y_pred_inputs[:, i_V]
    dVdx = y_pred_inputs[:, i_dx]
    dVdy = y_pred_inputs[:, i_dy]

    laplaceV_x1 = y_pred_inputs[:, i_ddx]
    laplaceV_x2 = y_pred_inputs[:, i_ddy]

    boundary_cond = y_true[:, i_V] < 998.

    diff_eq_loss = tf.square( laplaceV_x1 + laplaceV_x2 - y_true[:, i_ddx])
    #/ (tf.square( laplaceV_x1) + tf.square( laplaceV_x2) + 1e-4 )

    #boundary_loss =  ( tf.square( dVdx - y_true[:, i_dx] ) + tf.square( dVdy - y_true[:, i_dy] )   )

    boundary_loss =  1e5 * (tf.square( V - y_true[:, i_V] ) + tf.square( dVdx - y_true[:, i_dx] ) + tf.square( dVdy - y_true[:, i_dy] )   )

    loss_tensor = tf.where(boundary_cond, boundary_loss, diff_eq_loss)

    #loss_tensor = tf.where(x_cond, boundary_loss, tf.where(y_cond, boundary_loss, diff_eq_loss ))


    #loss_tensor = diff_eq_loss

    #print(y_true.shape)
    #print(y_pred_inputs.shape)

    #loss = tf.reduce_mean(loss_tensor)

    return loss_tensor

class GradModel(tfk.Model):
    """
    i_inputs_x = 0
    i_inputs_y = 1

    i_V = 2
    i_ddx = 5
    i_ddy = 6

    """

    def call(self, inputs, **kwargs):

        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(inputs)
            y_bar = super().call(inputs, **kwargs)

            grad_V = tape.gradient(y_bar[:, i_V], inputs)
            laplaceV_x1 = tape.gradient(grad_V[:, 0], inputs)[:, :1]
            laplaceV_x2 = tape.gradient(grad_V[:, 1], inputs)[:, 1:]

            if grad_V is None:
                raise Exception('Problem tracking dy_dx1: make sure tape.watch is applied and tensor is watchable.')

        dy_dx = tf.concat([laplaceV_x1, laplaceV_x2], axis=1)

        #print(grad_V.shape)
        print(dy_dx.shape)
        print(y_bar.shape)

        return tf.concat([y_bar, grad_V, dy_dx], axis=1)

def custom_model():


    submodel = tf.keras.Sequential([

        SinusodialRepresentationDense(64, activation='sine', w0=1.0),
        SinusodialRepresentationDense(64, activation='sine', w0=1.0),
        SinusodialRepresentationDense(64, activation='sine', w0=1.0),

        tfkl.Dense(1),
    ])

    return submodel


def run(epochs=20):
    """

    0   input x1
    1   input x2
    2   V
    3   d2V/dx1^2
    4   d2V/dx2^2

    """

    submodel = SIRENModel(units=128, final_units=1, final_activation='linear',
                   num_layers=4, w0=1.0, w0_initial=30.0)

    inputs = tfk.Input(shape=(2,))

    #optim = tfk.optimizers.Nadam(learning_rate=1e-3, decay=10)
    optim = tfk.optimizers.Adam(learning_rate=1e-4)
    #optim = tfk.optimizers.Nadam(learning_rate=1e-5, clipnorm=1.0)
    #optim = tfa.optimizers.MovingAverage(optim, start_step=weights_avg)

    #outputs = submodel(inputs)
    outputs = tfkl.Concatenate()([inputs, submodel(inputs)])


    model = GradModel(inputs=inputs, outputs=outputs)

    model.compile(
                optimizer=optim,
                loss=my_loss,       # mean squared error
                metrics=['mae'])  # mean absolute error

    #X = tf.random.uniform(shape=(100,1))
    #X = np.linspace(0.,1.,TIME_STEPS, dtype='float32')*PERIOD
    #X = np.concatenate([np.zeros(100), np.linspace(0., 1, 2000*BATCH_SIZE, dtype='float32')])
    #l = len(X)

    size = 1.
    n = 400
    X, y = make_X(n=n, size=size)

    img = make_neoval_image(n=n, value = 1e1)

    #img = init_y_image(n, init_val = 0.)
    #img = image_bounding_box(img, value = -0.2)
    #img = dipole(img, r=5e-2, d=3, value=1e2, method=circle_perimeter)
    #img = image_disk(img, int(n/2), int(n/2), int(5e-2*n), value=1e2, method=circle_perimeter)


    #r = 5e-2
    #img = image_ellipse_perimeter(img, r=-5e-2, c=5e-2, r_radius=r, c_radius=2*r, value=1e2)


    img_bb = init_y_image(n, init_val=999.)
    img_bb = image_bounding_box(img_bb, value=0.0)
    #img_bb = image_two_plates(img_bb, value=0.0)

    y[:, i_V] = img_bb.reshape(-1)

    y[:, i_dx] = img_bb.reshape(-1)
    y[:, i_dy] = img_bb.reshape(-1)

    y[:, i_ddx] = img.reshape(-1)

    #I_boundary = y[:, 1] < 998
    #X = X[I_boundary]
    #y = y[I_boundary]

    # upsample boundaries
    #I_boundary = y[:, 1] < 998
    #frac_boundary = I_boundary.sum()/len(y)
    #X_balanced =np.concatenate( [ X[I_boundary]]*int(1/ frac_boundary) + [X[~I_boundary]]  )
    #y_balanced =np.concatenate( [ y[I_boundary]]*int(1/ frac_boundary) + [y[~I_boundary]]  )

    #X = np.array([[0.2, 0.], [-0.2, 0.]], dtype='float32')

    X = tf.constant(X, dtype=tf.float32)

    #y = np.array([[0., -1., 0., 0., 0.], [0., 1., 0., 0., 0.]], dtype='float32')



    dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(len(X), reshuffle_each_iteration=True).repeat(50).batch(BATCH_SIZE).prefetch(1)

    #.map(lambda x, y :
    # (tf.random.uniform(shape=[BATCH_SIZE,2],minval=-1., maxval=1.), x,
    #  y )
    # )


    #dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(100000, reshuffle_each_iteration=True).take(100000).batch(BATCH_SIZE).prefetch#(1).map(lambda x, y :
    # (tf.concat([tf.random.uniform(shape=[BATCH_SIZE,2],minval=-1., maxval=1.), x], axis=0),
    #  tf.concat([999.*tf.ones(shape=[BATCH_SIZE,5]), y], axis=0) )
    # )


    #dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(BATCH_SIZE).prefetch(1)



    #X, y = make_X(n=1000)

    #dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.float32)))
    #.map(lambda x, y : tf.random.uniform(shape=[1],minval=-1e-2), y)


    hist = model.fit(dataset,
        epochs = epochs,
        #batch_size = BATCH_SIZE,
        verbose = 2
        )

    X_out, _ = make_X(n=n, size=size)
    #X_out = np.random.uniform(-1, 1, size=(10000,2))

    #print(model.predict(X_out))
    y_bar = model.predict(X_out)


    pd.DataFrame(y_bar).to_csv('tmp.csv', index=False)

    L_weights = model.get_weights()
    with open('weights.p', 'wb') as f:
        pickle.dump(L_weights, f)

    #y = np.exp(-0.1*X_out)
    #df_out = pd.DataFrame({'x' : X_out, 'y_bar' : y_bar.reshape(-1), 'y' : y})
    #df_out['e'] = abs(df_out['y_bar'] - df_out['y'])
    #print(df_out)
    #mean_error = df_out['e'].mean()
    #print(f'mean error {mean_error}')

    return



def make_arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', help="Epochs in training.", type = int, default = None)


    return parser

if __name__ == "__main__":

    # no args at the moment.
    parser = make_arg_parser()
    args = parser.parse_args()
    run( )
