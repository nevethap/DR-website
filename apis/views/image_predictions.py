from django.http import JsonResponse

from apis.utils.prediction_transformation_util import PredictionTransformationUtil
from retina.classifier import eval


class ImagePredictionsView:

    def getImages(self, request, start_index, end_index, filter_choice):
        test_labels, predictions, files = eval.run_evaluation(int(start_index), int(end_index), filter_choice)
        transformation_util = PredictionTransformationUtil()
        predictions_json = transformation_util.transform(test_labels, predictions, files)
        return JsonResponse(predictions_json, safe=False)