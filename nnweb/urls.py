from django.conf.urls import url

from apis.views.image_predictions import ImagePredictionsView

image_predictions = ImagePredictionsView()

urlpatterns = [
    url(r'^predictions/start_index=(\d+)/end_index=(\d+)/filter_choice=(.*)/',image_predictions.getImages,name = "predictions")
]
