import numpy as np
import sklearn.metrics as sm
from keras import backend as K

import retina.classifier.input as ip


def evaluate(start_index, end_index, X_test, Y_test, files_list, vgg16_model, top_model):
    print('Generating bottleneck features . . . ')

    bottleneck_features = vgg16_model.predict(X_test)

    print('Done!')

    predictions = top_model.predict_classes(bottleneck_features)

    return Y_test, predictions, files_list


def run_evaluation(start_index, end_index, filter_choice):
    X_test, Y_test, files_list = ip.read_images(start_index, end_index, filter_choice)
    vgg16_model = ip.load_vgg16_model()
    top_model = ip.load_top_model()
    test_labels, predictions, files = evaluate(start_index, end_index, X_test, Y_test, files_list, vgg16_model, top_model)
    return test_labels, predictions, files


if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    run_evaluation()
