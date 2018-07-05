import os
import numpy as np
import pandas as pd
import h5py
import random
from sklearn.metrics import confusion_matrix

from keras.regularizers import WeightRegularizer, ActivityRegularizer, l1, l2
#from keras.initializers import random_unifom
from keras.optimizers import RMSprop, Adam
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, core
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU, SReLU
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers.core import Flatten, Dense, Dropout, Permute, Activation, Reshape
from keras.objectives import mean_squared_error, categorical_crossentropy

from dask import delayed
import dask.array as da

try:
    import cPickle as pickle
except:
    import pickle
import time


def load_data_test(video_days, video_names, ann_orig, data_dir):
    """ This function creates test-images (no patches, entire images) -
    used in inference phase."""

    img_rows = 1152
    img_cols = 2880

    [x,y] = ann_orig.shape

    patch_list = []
    for video_day in video_days:
        for video_name in video_names:
                video = 'hsi_rc_{}_{}.h5'.format(video_day, video_name) #Name video
                img_path = os.path.join(data_dir, video) #Path to video
                f = h5py.File(img_path)
                dataset_name = [key for key in f.keys()][0]
                hsi_img_shape = f[dataset_name].shape
                f.close()

                row = 96
                col = 0
                patch_list += [{
                    'col_left': col,
                    'row_top': row,
                    'width': img_cols,
                    'height': img_rows,
                    'video_day': video_day,
                    'video_hour': video_name,
                    }]

    # make datafram list
    random.seed(1000)
    patch_list = random.sample(patch_list, len(patch_list))
    patch_list_df = pd.DataFrame(patch_list)
    patch_list_test = patch_list_df

    return patch_list_test

def load_data_description(img_rows, img_cols, nr_of_patches, video_days, video_names, ann_orig, data_dir,
                             classes, dir_exp, data_name, splitratio=0.2):

    '''This function loads all HS images (h5-files) and adds information about the patches
    of all images to patch_lists. One list for training and another one for validation.'''

    patch_list = []
    for video_day in video_days:
        for video_name in video_names:
                video = 'hsi_{}_{}.h5'.format(video_day, video_name) #Name video
                img_path = os.path.join(data_dir, video) #Path to video
                f = h5py.File(img_path)
                dataset_name = [key for key in f.keys()][0]
                [x_hsi, y_hsi, channels_hsi] = f[dataset_name].shape
                f.close()

                # random selection of coordinates top left point of patch
                [row_max, col_max] = ann_orig.shape
                row_max_sub = row_max - img_rows
                col_max_sub = col_max - img_cols

                for nr in range(0,nr_of_patches):
                    row_rand = random.randrange(0,row_max_sub,1)
                    col_rand = random.randrange(0,col_max_sub,1)
                    row = row_rand
                    col = col_rand
                    # only select patches if at least one class is present
                    if len(set(classes).intersection(set(np.unique(ann_orig[row:row + img_rows, col:col + img_cols])))) > 0:
                        # only select patches if more than xxx percent is annotated
                        seg = ann_orig[row:row + img_rows, col:col + img_cols]
                        [x_seg, y_seg] = seg.shape
                        count_0 = 0
                        for x in range(0,x_seg):
                            for y in range(0,y_seg):
                                if seg[x,y] == 0:
                                    count_0 = count_0 + 1
                        perc_0 = (count_0/(x_seg*y_seg))*100

                        classx = [int(x) for x in set(np.unique(ann_orig[row:row + img_rows, col:col + img_cols])) if x!=0]
                        patch_list += [{
                            'col_left': col,
                            'row_top': row,
                            'width': img_cols,
                            'height': img_rows,
                            'video_day': video_day,
                            'video_hour': video_name,
                            'class': classx
                        }]

    # shuffle list and make list dataframe
    random.seed(1000)
    patch_list = random.sample(patch_list, len(patch_list))
    patch_list_df = pd.DataFrame(patch_list)

    # split list in train and test set
    split = int(splitratio*len(patch_list))
    patch_list_val = patch_list_df[:split]
    patch_list_train = patch_list_df[split:]

    return patch_list_train, patch_list_val, patch_list_df



def batch_function(patch_list, ann_orig, data_dir, classes):
    '''Returns X_data (HS image patches), Y_data (mask patches) based on list of patches,
    used as input for neural network'''

    masks = []
    hsi_patches = []
    files = []
    for _, patch in patch_list.iterrows():
        video = 'hsi_{}_{}.h5'.format(patch.video_day, patch.video_hour)
        img_path = os.path.join(data_dir, video)
        f = h5py.File(img_path)
        files.append(f)
        dataset_name = [key for key in f.keys()][0]
        hsi_img = da.from_array(f[dataset_name], chunks=(200, 200, 25))
        hsi_img = hsi_img.transpose([2, 0, 1]) # model needs input (#channels, heigth, width)

        patch_array = hsi_img[:, patch.row_top:patch.row_top + patch.height, patch.col_left:patch.col_left + patch.width]
        d = delayed(np.pad)(patch_array, ((0, 0), (0, patch.height - patch_array.shape[1]), (0, patch.width - patch_array.shape[2])), mode='constant')
        hsi_patches.append(da.from_delayed(d, shape=(25, patch.height, patch.width)))

        ann = ann_orig[:hsi_img.shape[1], :hsi_img.shape[2]]
        mask = ann[patch.row_top:patch.row_top + patch.height, patch.col_left:patch.col_left + patch.width]
        mask = np.pad(mask, ((0, patch.height - mask.shape[0]), (0, patch.width - mask.shape[1])), mode='constant')
        mask_arrays = (mask[np.newaxis, :, :] == np.array(classes)[:, np.newaxis, np.newaxis]).astype(int)
        masks.append(mask_arrays)

    X_data = da.stack(hsi_patches).compute()
    Y_data = np.asarray(masks)

    for f in files:
        f.close()

    return X_data, Y_data


def fscore(y_true, y_pred):
    y_true_2D = K.max(y_true, axis=1, keepdims=False)
    y_pred_2D = K.max(y_true*y_pred, axis=1, keepdims=False)

    smooth = 1.
    y_true_f = K.flatten(y_true_2D)
    y_pred_f = K.flatten(y_pred_2D)
    intersection = K.sum(y_true_f * y_pred_f)
    f = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    return f

def overall_acc(y_true, y_pred):
    y_true_2D = K.max(y_true, axis=1, keepdims=False)
    y_pred_2D = K.max(y_true*y_pred, axis=1, keepdims=False)

    y_true_f = K.sum(K.flatten(y_true_2D))
    y_pred_f = K.sum(K.flatten(y_pred_2D))

    acc = y_pred_f / (y_true_f + 0.0001)

    return acc

def custom_loss(weights=None):
    def custom_loss_test(y_true, y_pred):
        if weights is not None:
            weights_tensor = np.array(weights)[:, np.newaxis, np.newaxis]
            w_tensor = weights_tensor * K.ones_like(y_true)
            cover_pred = y_true * y_pred
            minimize =  (y_true - cover_pred)*w_tensor
        else:
            cover_pred = y_true * y_pred
            minimize = y_true - cover_pred

        loss = K.mean(K.square(minimize))
        return loss

    return custom_loss_test

def custom_crossentropy(weights=None):
    def cr_loss_masked(y_true, y_pred):
        mask = K.sum(y_true, axis=0)
        y_pred_mask = mask * y_pred #only remove pixels without annotations
        loss = K.categorical_crossentropy(y_true, y_pred_mask)
        return loss

    return cr_loss_masked


def make_unet_model(optimizer, img_rows, img_cols, classes, dropoutrate, weights=None):
    wreg = None
    ac = 'elu'
    areg = 'None'
    win = 'glorot_normal'

    inputs = Input((25, img_rows, img_cols))
    conv1 = BatchNormalization(axis=1, epsilon=0.001)(inputs)
    conv1 = Convolution2D(32, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv1)
    conv1 = Convolution2D(32, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv1)
    conv1 = Dropout(dropoutrate)(conv1)
    #conv1 = BatchNormalization(axis=1)(conv1)

    conv2 = MaxPooling2D(pool_size=(2,2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv2)
    #conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Convolution2D(64, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv2)
    conv2 = Dropout(dropoutrate)(conv2)
    #conv2 = BatchNormalization(axis=1)(conv2)

    conv3 = MaxPooling2D(pool_size=(2,2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv3)
    #conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Convolution2D(128, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv3)
    conv3 = Dropout(dropoutrate)(conv3)
    #conv3 = BatchNormalization(axis=1)(conv3)

    conv4 = MaxPooling2D(pool_size=(2,2))(conv3)
    conv4 = Convolution2D(256, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv4)
    #conv4 = BatchNormalization(axis=1)(conv4)
    conv4 = Convolution2D(256, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv4)
    conv4 = Dropout(dropoutrate)(conv4)
    #conv4 = BatchNormalization(axis=1)(conv4)

    # deepest layer (5th convolutional layer)
    conv5 = MaxPooling2D(pool_size=(2,2))(conv4)
    conv5 = Convolution2D(512, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv5)
    #conv5 = BatchNormalization(axis=1)(conv5)
    conv5 = Convolution2D(512, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv5)
    conv5 = Dropout(dropoutrate)(conv5)
    #conv5 = BatchNormalization(axis=1)(conv5)

    up6 = UpSampling2D(size=(2,2))(conv5)
    conv6 = merge([up6, conv4], mode='concat', concat_axis=1)
    conv6 = Dropout(dropoutrate)(conv6)
    conv6 = Convolution2D(256, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv6)
    conv6 = Convolution2D(256, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv6)

    up7 = UpSampling2D(size=(2,2))(conv6)
    conv7 = merge([up7, conv3], mode='concat', concat_axis=1)
    conv7 = Dropout(dropoutrate)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv7)
    conv7 = Convolution2D(128, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv7)

    up8 = UpSampling2D(size=(2,2))(conv7)
    conv8 = merge([up8, conv2], mode='concat', concat_axis=1)
    conv8 = Dropout(dropoutrate)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv8)
    conv8 = Convolution2D(64, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv8)

    up9 = UpSampling2D(size=(2,2))(conv8)
    conv9 = merge([up9, conv1], mode='concat', concat_axis=1)
    conv9 = Dropout(dropoutrate)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv9)
    conv9 = Convolution2D(32, 3, 3, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv9)

    conv11 = Convolution2D(len(classes), 1, 1, activation=ac, border_mode='same', init=win, W_regularizer=wreg)(conv9) #batch x 5 x 1 x 1
    conv11 = core.Reshape((len(classes),img_rows*img_cols))(conv11) # batch x 5 x nr
    conv11 = core.Permute((2,1))(conv11) # batch x nr x 5

    conv12 = core.Activation('softmax')(conv11) # batch x nr x 5

    conv13 = core.Permute((2,1))(conv12) #batch x 5 x nr
    conv13 = core.Reshape((len(classes),img_rows,img_cols))(conv13)

    model = Model(input=inputs, output=conv13)
    model.compile(optimizer=optimizer, loss=custom_loss(weights), metrics = [fscore, overall_acc])

    return model
