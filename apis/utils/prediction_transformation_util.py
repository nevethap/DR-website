import numpy as np

class PredictionTransformationUtil:
    def transform(self, test_labels, predictions, files):
        transformed_predictions = []
        prediction_dict = {}
        for label, predicted_res, image_name in zip(test_labels, predictions, files):
            prediction_dict['actual'] = np.asscalar(label)
            prediction_dict['predicted'] = np.asscalar(predicted_res[0])
            prediction_dict['img_url'] = image_name

            transformed_predictions.append(prediction_dict)
        return transformed_predictions
