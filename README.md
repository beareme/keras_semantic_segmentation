# Keras Semantic Segmentation

Fully Convolutional Networks for Semantic Segmentation in Keras

## Usage

 - Modify the file train.sh including the paths to multimodal_keras_wrapper library and Keras.
 - Prepare the dataset following the same format as sample_data.
 - Insert the data paths in config.py and modify any parameter as desired.
 - Run ./train.sh to train a model.

Note that the code has been tested using Theano as backend.

## Dependencies

The following dependencies are required for using this library:
 - [Custom Keras fork](https://github.com/MarcBS/keras/releases/tag/2.0.9) >= 2.0.9
 - [Multimodal Keras Wrapper](https://github.com/MarcBS/multimodal_keras_wrapper/releases/tag/v2.1.6) >= 2.1.6

## Download

You can donwload a [zip file of the source code](https://github.com/beareme/keras_semantic_segmentation/archive/master.zip) directly.

Alternatively, you can clone it from GitHub as follows:
```
git clone https://github.com/beareme/keras_semantic_segmentation.git
```

## Keras

For additional information on the Deep Learning library, visit the official web page www.keras.io or the GitHub repository https://github.com/fchollet/keras.

## References

S. Jégou, M. Drozdzal, D. Vazquez, A. Romero, Y. Bengio (2017). The One Hundred Layers Tiramisu: Fully Convolutional Densenets for Semantic Segmentation. IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 1175-1183.

O. Ronneberger, P. Fischer, T. Brox (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 234-241.
