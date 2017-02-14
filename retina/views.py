from django.views import generic
from keras import backend as K

from retina.classifier import eval
from .models import Image


def load_results(start_index, end_index, filter_choice):
    K.set_image_dim_ordering('th')
    test_data, test_labels, predictions, files = eval.run_evaluation(start_index, end_index, filter_choice)

    for data, label, prediction, file in zip(test_data, test_labels, predictions, files):
        Image(srcFile=file, actual=label, preidcted=prediction, img_logo=file).save()


class IndexView(generic.TemplateView):
    template_name = 'retina/index.html'
    # context_object_name = 'all_images'
    #
    # def get_queryset(self):
    #     Image.objects.all().delete()
    #     load_results()
    #     return Image.objects.all()


class ListView(generic.ListView):
    template_name = 'retina/list.html'
    context_object_name = 'all_images'

    def get_queryset(self):
        Image.objects.all().delete()
        load_results(20, 30, 'diseased')
        return Image.objects.all()


class DetailView(generic.DetailView):
    model = Image
    template_name = 'retina/detail.html'
