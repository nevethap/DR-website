import numpy as np

class PredictionTransformationUtil:
    def transform(self, test_labels, predictions, files, precision, recall):
        predictions_all = {}
        predictions_all['predictions'] = []
        for label, predicted_res, image_name in zip(test_labels, predictions, files):
            prediction_dict = {}
            prediction_dict['actual'] = np.asscalar(label)
            prediction_dict['predicted'] = np.asscalar(predicted_res[0])
            prediction_dict['img_url'] = image_name
            predictions_all['predictions'].append(prediction_dict)
        predictions_all['precision'] = precision # TODO: remove hardcoding with actual values
        predictions_all['recall'] = recall # TODO: remove hardcoding with actual values
        return predictions_all
