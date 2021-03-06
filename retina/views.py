from django.views import generic
from keras import backend as K

from retina.classifier import eval
from .models import Image


def load_results():
    K.set_image_dim_ordering('th')
    test_data, test_labels, predictions, files = eval.run_evaluation()
    for data, label, prediction, file in zip(test_data, test_labels, predictions, files):
        Image(srcFile=file, actual=label, preidcted=prediction, img_logo=file).save()


class IndexView(generic.ListView):
    template_name = 'retina/index.html'
    context_object_name = 'all_images'

    def get_queryset(self):
        Image.objects.all().delete()
        load_results()
        return Image.objects.all()


class DetailView(generic.DetailView):
    model = Image
    template_name = 'retina/detail.html'
