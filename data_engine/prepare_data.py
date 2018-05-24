from keras_wrapper.dataset import Dataset, saveDataset, loadDataset
from utils.common import get_num_captions
import numpy as np
import logging

def build_dataset(params):
    
    if params['REBUILD_DATASET']: # We build a new dataset instance
        if(params['VERBOSE'] > 0):
            silence=False
            logging.info('Building ' + params['DATASET_NAME'] + ' dataset')
        else:
            silence=True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME']
        ds = Dataset(name, base_path, silence=silence)

        ##### INPUT DATA
        # Let's load the images (inputs)

        ### IMAGES
        list_train = base_path + '/' + params['IMG_FILES']['train'][0]
        ds.setInput(list_train, 'train',
                    type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                    img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                    use_RGB=params['RGB'])
        if 'val' in params['IMG_FILES'] and params['IMG_FILES']['val']:
            list_val = base_path + '/' + params['IMG_FILES']['val'][0]
            ds.setInput(list_val, 'val',
                        type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                        img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                        use_RGB=params['RGB'])
        if 'test' in params['IMG_FILES'] and params['IMG_FILES']['test']:
            list_test = base_path + '/' + params['IMG_FILES']['test'][0]
            ds.setInput(list_test, 'test',
                        type='raw-image', id=params['INPUTS_IDS_DATASET'][0],
                        img_size=params['IMG_SIZE'], img_size_crop=params['IMG_CROP_SIZE'],
                        use_RGB=params['RGB'])

        # Train mean
        if params['MEAN_IMAGE']:
            # if params['NORMALIZE']:
            #    params['MEAN_IMAGE'] = [m / 255. for m in params['MEAN_IMAGE']]
            ds.setTrainMean(params['MEAN_IMAGE'], params['INPUTS_IDS_DATASET'][0])
        else:
            ds.calculateTrainMean(params['INPUTS_IDS_DATASET'][0])

        ##### OUTPUT DATA
        if params['TYPE_OUT'] == '3DLabel':
            # Set list of classes (strings)
            ds.setClasses(base_path + '/' + params['CLASSES_PATH'], params['OUTPUTS_IDS_DATASET'][0])
        elif params['TYPE_OUT'] == '3DSemanticLabel':
            # Set list of classes (strings)
            classes_names = []
            with open(base_path + '/' + params['CLASSES_PATH'], 'r') as file:
                for line in file:
                    line = line.rstrip('\n').split(',')[0]
                    classes_names.append(line)
            ds.setClasses(classes_names, params['OUTPUTS_IDS_DATASET'][0])
            ds.setSemanticClasses(base_path + '/' + params['CLASSES_PATH'], params['OUTPUTS_IDS_DATASET'][0])


        ### 3DLabels or 3DSemanticLabels
        ds.setOutput(base_path+'/'+params['IMG_FILES']['train'][1], 'train',
                   type=params['TYPE_OUT'], id=params['OUTPUTS_IDS_DATASET'][0],
                   associated_id_in=params['INPUTS_IDS_DATASET'][0], num_poolings=params['NUM_MODEL_POOLINGS'])
        if 'val' in params['IMG_FILES'] and params['IMG_FILES']['val']:
            ds.setOutput(base_path+'/'+params['IMG_FILES']['val'][1], 'val',
                       type=params['TYPE_OUT'], id=params['OUTPUTS_IDS_DATASET'][0],
                       associated_id_in=params['INPUTS_IDS_DATASET'][0], num_poolings=params['NUM_MODEL_POOLINGS'])
        if 'test' in params['IMG_FILES'] and params['IMG_FILES']['test']:
            ds.setOutput(base_path+'/'+params['IMG_FILES']['test'][1], 'test',
                       type=params['TYPE_OUT'], id=params['OUTPUTS_IDS_DATASET'][0],
                       associated_id_in=params['INPUTS_IDS_DATASET'][0], num_poolings=params['NUM_MODEL_POOLINGS'])

        if params['DISCARD_CLASSES']:
            weights = np.ones((params['NUM_CLASSES'],))
            for c in params['DISCARD_CLASSES']:
                weights[c] = 0.0
            ds.extra_variables['class_weights_' + params['OUTPUTS_IDS_DATASET'][0]] = weights

        if params['WEIGHT_CLASSES']:
            weights = params['WEIGHT_CLASSES']
            ds.extra_variables['class_weights_' + params['OUTPUTS_IDS_DATASET'][0]] = weights

        ### Single multi-label
        if params['APPLY_MULTILABEL_CLASSIFICATION']:
            n_classes = len(ds.classes[params['OUTPUTS_IDS_DATASET'][0]])
            multilabel = convert3DLabels2multilabel(base_path+'/'+params['IMG_FILES']['train'][1], n_classes)
            ds.setOutput(multilabel, 'train', type='binary', id=params['OUTPUTS_IDS_DATASET'][1])
            if 'val' in params['IMG_FILES'] and params['IMG_FILES']['val']:
                multilabel = convert3DLabels2multilabel(base_path + '/' + params['IMG_FILES']['val'][1], n_classes)
                ds.setOutput(multilabel, 'val', type='binary', id=params['OUTPUTS_IDS_DATASET'][1])
            if 'test' in params['IMG_FILES'] and params['IMG_FILES']['test']:
                multilabel = convert3DLabels2multilabel(base_path + '/' + params['IMG_FILES']['test'][1], n_classes)
                ds.setOutput(multilabel, 'test', type='binary', id=params['OUTPUTS_IDS_DATASET'][1])


        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])
    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH']+'/Dataset_'+params['DATASET_NAME']+'.pkl')

    return ds
