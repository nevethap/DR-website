import numpy as np
import sklearn.metrics as sm
from keras import backend as K

import retina.classifier.input as ip


def evaluate(X_test, Y_test, files_list, vgg16_model, top_model):
    test_data = []
    test_data.extend(X_test[0:10])
    test_data.extend(X_test[100:110])
    test_data = np.asarray(test_data)
    print(len(test_data))
    test_labels = np.hstack((Y_test[0:10], Y_test[100:110]))
    print(test_labels)

    files0 = files_list[0:10]
    files1 = files_list[100:110]
    files = files0 + files1

    print('Generating bottleneck features . . . ')

    bottleneck_features = vgg16_model.predict(test_data)

    print('Done!')

    predictions = top_model.predict_classes(bottleneck_features)

    print('Actual')
    print(test_labels)
    print('Predictions')
    print(np.concatenate(predictions))
    print('Recall : ' + str(sm.recall_score(test_labels, predictions)))
    print('Precision : ' + str(sm.precision_score(test_labels, predictions)))

    for data, label, prediction, file in zip(test_data, test_labels, predictions, files):
        print(str(label) + ' ' + str(prediction) + ' ' + file)

    return test_data, test_labels,  predictions, files


def run_evaluation():
    X_test, Y_test, files_list = ip.input_images()
    vgg16_model = ip.load_vgg16_model()
    top_model = ip.load_top_model()
    test_data, test_labels, predictions, files = evaluate(X_test, Y_test, files_list, vgg16_model, top_model)
    return test_data, test_labels, predictions, files


if __name__ == '__main__':
    K.set_image_dim_ordering('th')
    run_evaluation()
