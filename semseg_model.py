import keras
from keras.layers import Input, Dense, MaxPooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation, Permute
from keras.layers.core import Dropout, Dense, Flatten, Activation, Lambda, WeightedMerge
from keras.layers.convolutional import UpSampling2D
from keras.models import model_from_json, Model
from keras.layers.advanced_activations import PReLU
from keras.layers.normalization import BatchNormalization, L2_norm, signed_sqrt
from keras.optimizers import Adam, RMSprop, Nadam, Adadelta
from keras import backend as K
from keras.regularizers import l2

if int(keras.__version__.split('.')[0]) == 1:
    from keras.layers import Convolution2D as Conv2D
else:
    from keras.layers import Conv2D

from keras_wrapper.cnn_model import Model_Wrapper

import numpy as np
import os
import logging
import shutil
import time
import sys
import copy
sys.setrecursionlimit(1500)

class Segmentation_Model(Model_Wrapper):
    
    def __init__(self, params, type='VGG19', verbose=1, structure_path=None, weights_path=None,
                 model_name=None, vocabularies=None, store_path=None):
        """
            CNN_Model object constructor.

            :param params: all hyperparameters of the model.
            :param type: network name type (corresponds to any method defined in the section 'PREDEFINED MODELS' of this class). Only valid if 'structure_path' == None.
            :param verbose: set to 0 if you don't want the model to output informative messages
            :param structure_path: path to a Keras' model json file. If we speficy this parameter then 'type' will be only an informative parameter.
            :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
            :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
            :param vocabularies: vocabularies used for GLOVE word embedding
            :param store_path: path to the folder where the temporal model packups will be stored

            References:
                [PReLU]
                Kaiming He et al. Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

                [BatchNormalization]
                Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        """
        super(self.__class__, self).__init__(type=type, model_name=model_name,
                                             silence=verbose == 0, models_path=store_path, inheritance=True)

        self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']

        self.verbose = verbose
        self._model_type = type
        self.params = params
        self.vocabularies = vocabularies

        # Sets the model name and prepares the folders for storing the models
        self.setName(model_name, store_path=store_path)

        # Prepare model
        if structure_path:
            # Load a .json model
            if self.verbose > 0:
                logging.info("<<< Loading model structure from file "+ structure_path +" >>>")
            self.model = model_from_json(open(structure_path).read())
        else:
            # Build model from scratch
            if hasattr(self, type):
                if self.verbose > 0:
                    logging.info("<<< Building "+ type +" Model >>>")
                eval('self.'+type+'(params)')
            else:
                raise Exception('CNN_Model type "'+ type +'" is not implemented.')

        # Load weights from file
        if weights_path:
            if self.verbose > 0:
                logging.info("<<< Loading weights from file "+ weights_path +" >>>")
            self.model.load_weights(weights_path)

        # Print information of self
        if verbose > 0:
            print str(self)
            self.model.summary()

        self.setOptimizer()

    def setOptimizer(self):
        """
            Sets a new optimizer for the model.
        """

        super(self.__class__, self).setOptimizer(lr=self.params['LR'],
                                                 loss=self.params['LOSS'],
                                                 optimizer=self.params['OPTIMIZER'],
                                                 epsilon=self.params['EPSILON'],
                                                 sample_weight_mode='temporal' if self.params.get('SAMPLE_WEIGHTS', False) else None)



    def setName(self, model_name, store_path=None, clear_dirs=True):
        """
            Changes the name (identifier) of the Translation_Model instance.
        """
        if model_name is None:
            self.name = time.strftime("%Y-%m-%d") + '_' + time.strftime("%X")
            create_dirs = False
        else:
            self.name = model_name
            create_dirs = True

        if store_path is None:
            self.model_path = 'Models/' + self.name
        else:
            self.model_path = store_path


        # Remove directories if existed
        if clear_dirs:
            if os.path.isdir(self.model_path):
                shutil.rmtree(self.model_path)

        # Create new ones
        if create_dirs:
            if not os.path.isdir(self.model_path):
                os.makedirs(self.model_path)

    # ------------------------------------------------------- #
    #       VISUALIZATION: Methods for visualization
    # ------------------------------------------------------- #

    def __str__(self):
        """
            Plot basic model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t'+class_name +' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'MODEL PARAMETERS:\n'
        obj_str += str(self.params)
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str

    # ------------------------------------------------------- #
    #       PREDEFINED MODELS
    # ------------------------------------------------------- #

    def DebugModel(self, params):

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]
        
        inp = Input(shape=tuple([params['IMG_CROP_SIZE'][-1]]+params['IMG_CROP_SIZE'][:2]), name=self.ids_inputs[0])
        x = Conv2D(params['NUM_CLASSES'], (1, 1), padding='same') (inp)
        x = Reshape((params['NUM_CLASSES'],
                     params['IMG_CROP_SIZE'][0] * params['IMG_CROP_SIZE'][1]))(x)
        x = Permute((2, 1))(x)

        matrix_out = Activation(params['CLASSIFIER_ACTIVATION'], name=self.ids_outputs[0])(x)
        self.model = Model(inputs=inp, outputs=matrix_out)
        
        
    def ClassicUpsampling(self, params):
        """
            References:
                Olaf Ronneberger et al. U-net: Convolutional networks for biomedical image segmentation.
        """
        
        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        crop = params['IMG_CROP_SIZE']
        image = Input(shape=tuple([crop[-1], None, None]), name=self.ids_inputs[0])

        # Downsampling path and recover skip connections for each transition down block
        concat_axis = 1

        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(image)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Middle of the path (bottleneck)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        # Upsampling path
        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
        up_conv5 = ZeroPadding2D()(up_conv5)
        up6 = Concat(cropping=[None, None, 'center', 'center'])([conv4, up_conv5])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
        up_conv6 = ZeroPadding2D()(up_conv6)
        up7 = Concat(cropping=[None, None, 'center', 'center'])([conv3, up_conv6])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        up_conv7 = ZeroPadding2D()(up_conv7)
        up8 = Concat(cropping=[None, None, 'center', 'center'])([conv2, up_conv7])
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        up_conv8 = ZeroPadding2D()(up_conv8)
        up9 = Concat(cropping=[None, None, 'center', 'center'])([conv1, up_conv8])
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        # Final classification layer (batch_size, classes, width, height)
        x = Conv2D(params['NUM_CLASSES'], (1, 1), border_mode='same')(conv9)

        # Reshape to (None, width*height, classes) before applying softmax
        x = Lambda(lambda x: x.flatten(ndim=3), output_shape=lambda s: tuple([s[0], params['NUM_CLASSES'], None]))(x)
        x = Permute((2, 1))(x)

        matrix_out = Activation(params['CLASSIFIER_ACTIVATION'], name=self.ids_outputs[0])(x)
        self.model = Model(inputs=image, outputs=matrix_out)


    def Tiramisu(self, params):
        """
            References:
                Simon Jegou et al. The One Hundred Layers Tiramisu: Fully Convolutional Densenets for Semantic Segmentation.
        """

        self.ids_inputs = params["INPUTS_IDS_MODEL"]
        self.ids_outputs = params["OUTPUTS_IDS_MODEL"]

        crop = params['IMG_CROP_SIZE']
        image = Input(shape=tuple([crop[-1], None, None]), name=self.ids_inputs[0])

        init_filters = params['TIRAMISU_INIT_FILTERS']
        growth_rate = params['TIRAMISU_GROWTH_RATE']
        num_transition_blocks = params['TIRAMISU_N_TRANSITION_BLOCKS']

        n_layers_down = params['TIRAMISU_N_LAYERS_DOWN']
        n_layers_up = params['TIRAMISU_N_LAYERS_UP']
        bottleneck_layers = params['TIRAMISU_BOTTLENECK_LAYERS']

        # Downsampling path and recover skip connections for each transition down block
        x = Conv2D(init_filters, (3, 3), kernel_initializer=params['WEIGHTS_INIT'], padding='same', name='conv_initial')(image)
        prev_filters = init_filters

        skip_conn = []
        for td in range(num_transition_blocks):
            nb_filters_conv = prev_filters + n_layers_down[td] * growth_rate
            [x, skip] = self.add_transitiondown_block(x, nb_filters_conv,
                                                      2, params['WEIGHTS_INIT'],
                                                      n_layers_down[td], growth_rate,
                                                      params['DROPOUT_P'])
            skip_conn.append(skip)
            prev_filters = nb_filters_conv

        # Middle of the path (bottleneck)
        x = self.add_dense_block(x, bottleneck_layers, growth_rate, params['DROPOUT_P'], params['WEIGHTS_INIT'])  # feature maps: 512 input, 592 output

        # Upsampling path
        skip_conn = skip_conn[::-1]
        prev_filters = bottleneck_layers*growth_rate
        for tu in range(num_transition_blocks):
            x = self.add_transitionup_block(x, skip_conn[tu],
                                            prev_filters, params['WEIGHTS_INIT'],
                                            n_layers_up[tu], growth_rate, params['DROPOUT_P'])
            prev_filters = n_layers_up[tu]*growth_rate

        # Final classification layer (batch_size, classes, width, height)
        x = Conv2D(params['NUM_CLASSES'], (1, 1), kernel_initializer=params['WEIGHTS_INIT'], padding='same')(x)

        # Reshape to (None, width*height, classes) before applying softmax
        x = Lambda(lambda x: x.flatten(ndim=3), output_shape=lambda s: tuple([s[0], params['NUM_CLASSES'], None]))(x)
        x = Permute((2, 1))(x)

        matrix_out = Activation(params['CLASSIFIER_ACTIVATION'], name=self.ids_outputs[0])(x)
        self.model = Model(inputs=image, outputs=matrix_out)
