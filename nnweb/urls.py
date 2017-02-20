from django.conf.urls import url

from apis.views.image_predictions import ImagePredictionsView

from apis.views.images import Images

images = Images()

image_predictions = ImagePredictionsView()

urlpatterns = [
    url(r'^getImages/', images.getImages, name = 'getImages'),
    url(r'^predictions/start_index=(\d+)/end_index=(\d+)/filter_choice=(.*)/',image_predictions.getImages,name = 'predictions')
]
