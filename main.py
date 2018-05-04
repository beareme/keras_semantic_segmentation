import logging
from timeit import default_timer as timer

from data_engine.prepare_data import build_dataset
from semseg_model import Segmentation_Model

from keras_wrapper.cnn_model import loadModel, saveModel
from keras_wrapper.utils import *
from keras_wrapper.extra.callbacks import EvalPerformance
from keras_wrapper.extra.evaluation import select as selectMetric
from keras_wrapper.extra.read_write import list2file

from keras.layers.attention import AttentionComplex
from keras.layers.core import WeightedMerge

import sys
import ast
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def train_model(params):
    """
        Main function
    """

    if(params['RELOAD'] > 0):
        logging.info('Resuming training.')

    check_params(params)

    ########### Load data
    dataset = build_dataset(params)

    # Keep original images size if IMAGE_CROPPING == False
    if not params['IMAGE_CROPPING']:
        dataset.img_size_crop = dataset.img_size
    ###########

    ########### Build model
    if(params['RELOAD'] == 0): # build new model 
        cnn_model = Segmentation_Model(params, type=params['MODEL_TYPE'], verbose=params['VERBOSE'],
                                model_name=params['MODEL_NAME'],
                                store_path=params['STORE_PATH'])

        # Define the inputs and outputs mapping from our Dataset instance to our model
        cnn_model.setInputsMapping(params['INPUTS_MAPPING'])
        cnn_model.setOutputsMapping(params['OUTPUTS_MAPPING'])

        # Save initial untrained model and try to load it again
        saveModel(cnn_model, 0)
        cnn_model=loadModel(params['STORE_PATH'], 0,
                              custom_objects={"AttentionComplex": AttentionComplex, 'WeightedMerge': WeightedMerge})
        cnn_model.params = params
        cnn_model.setOptimizer()

    else: # resume from previously trained model
        cnn_model = loadModel(params['STORE_PATH'], params['RELOAD'],
                              custom_objects={"AttentionComplex": AttentionComplex})
        cnn_model.model_path = params['STORE_PATH']
        cnn_model.params = params
        cnn_model.setOptimizer()
    ###########

    # Test model save/load
    saveModel(cnn_model, 0)
    cnn_model = loadModel(params['STORE_PATH'], 0,
                          custom_objects={"AttentionComplex": AttentionComplex})
    cnn_model.setOptimizer()
    
    ########### Callbacks
    callbacks = buildCallbacks(params, cnn_model, dataset)
    ###########

    ########### Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'], 'batch_size': params['BATCH_SIZE'],
                       'lr_decay': params['LR_DECAY'], 'lr_gamma': params['LR_GAMMA'],
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'], 'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'], 'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks, 'reload_epoch': params['RELOAD'], 'epoch_offset': params['RELOAD'],
                       'data_augmentation': params['DATA_AUGMENTATION'], 'shuffle': params['SHUFFLE_TRAIN'],
                       'patience': params['PATIENCE'], 'metric_check': params['STOP_METRIC'], 'patience_check_split': params['PATIENCE_SPLIT'],
                       'normalize': params['NORMALIZE'], 'normalization_type': params['NORMALIZATION_TYPE'], 'mean_substraction': params['MEAN_SUBSTRACTION'],
                       'class_weights': params['OUTPUTS_IDS_DATASET'][0] if params['DISCARD_CLASSES'] or params['WEIGHT_CLASSES'] else None,
                        }

    cnn_model.trainNet(dataset, training_params)

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference, time_difference / 60.0))
    ###########
    

def apply_model(params):
    """
        Function for using a previously trained model for sampling.
    """
    
    ########### Load data
    dataset = build_dataset(params)

    # Keep original images size if IMAGE_RESIZE == False
    if not params['IMAGE_CROPPING']:
        dataset.img_size_crop = dataset.img_size
    ###########
    
    ########### Load model
    model = loadModel(params['STORE_PATH'], params['RELOAD'],
                            custom_objects={"AttentionComplex": AttentionComplex})
    model.setOptimizer()
    ###########

    ########### Apply sampling
    callbacks = buildCallbacks(params,model,dataset)
    callbacks[0].evaluate(params['RELOAD'], 'epoch')


def buildCallbacks(params, model, dataset):
    """
        Builds the selected set of callbacks run during the training of the model
    """

    callbacks = []

    if params['METRICS']:
        # Evaluate model
        extra_vars = {'n_parallel_loaders': params['PARALLEL_LOADERS']}
        extra_vars['n_classes'] = len(dataset.dic_classes[params['OUTPUTS_IDS_DATASET'][0]].keys())
        extra_vars['discard_classes'] = params['DISCARD_CLASSES']

        callback_metric = EvalPerformance(model, dataset,
                                          gt_id=params['OUTPUTS_IDS_DATASET'][0],
                                          metric_name=params['METRICS'],
                                          set_name=params['EVAL_ON_SETS'],
                                          batch_size=params['BATCH_SIZE'],
                                          each_n_epochs=params['EVAL_EACH'],
                                          extra_vars=extra_vars,
                                          normalize=params['NORMALIZE'],
                                          reload_epoch=params['RELOAD'],
                                          save_path=model.model_path,
                                          start_eval_on_epoch=params['START_EVAL_ON_EPOCH'],
                                          write_samples=params['WRITE_VALID_SAMPLES'],
                                          write_type='3DSemanticLabel',
                                          #is_3DLabel=is_3DLabel,
                                          input_id=params['INPUTS_IDS_DATASET'][0],
                                          verbose=params['VERBOSE'],
                                          eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                          eval_orig_size=params['EVAL_ORIG_SIZE'],
                                          output_types=params['TYPE_OUT'])

        callbacks.append(callback_metric)

    return callbacks


def check_params(params):
    if 'Glove' in params['MODEL_TYPE'] and params['GLOVE_VECTORS'] is None:
        logger.warning("You set a model that uses pretrained word vectors but you didn't specify a vector file."
                       "We'll train WITHOUT pretrained embeddings!")
    if params["USE_DROPOUT"] and params["USE_BATCH_NORMALIZATION"]:
        logger.warning("It's not recommended to use both dropout and batch normalization")


if __name__ == "__main__":

    # Use 'config_file' command line parameter for changing the name of the config file used
    cf = 'config'
    for arg in sys.argv[1:]:
        k, v = arg.split('=')
        if k == 'config_file':
            cf = v
    cf = __import__(cf)
    params = cf.load_parameters()

    try:
        for arg in sys.argv[1:]:
            k, v = arg.split('=')
            if k != 'config_file':
                params[k] = ast.literal_eval(v)
    except:
        print 'Overwritten arguments must have the form key=Value'
        exit(1)

    if(params['MODE'] == 'training'):
        logging.info('Running training.')
        train_model(params)
    elif(params['MODE'] == 'predicting'):
        logging.info('Running prediction.')
        apply_model(params)

    logging.info('Done!')   
