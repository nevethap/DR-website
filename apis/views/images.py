import os
from scipy import ndimage
from django.http import JsonResponse
import pandas as pd
import numpy as np


class Images:
    def __getImageUrls(self):
        img_path='retina/static/retina/retina_images/'
        labels = pd.read_csv('retina/classifier/labels/labels_for_class0_and_class1.csv', header=0)
        healthy_inps = []
        diseased_inps = []
        images = {}
        for file in os.listdir(img_path):
            file_name = file.replace('.jpeg', '')
            if file[:1] != '.':
                healthy_inp = {}
                diseased_inp = {}
                label = np.asscalar(labels[labels['image'] == file_name]['level'].values[0])
                if(label == 0):
                    healthy_inp['url'] = file
                    healthy_inp['actual'] = label
                    healthy_inps.append(healthy_inp)
                else:
                    diseased_inp['url'] = file
                    diseased_inp['actual'] = label
                    diseased_inps.append(diseased_inp)
        images['healthy'] = healthy_inps
        images['diseased'] = diseased_inps
        return images
    def getImages(self, request):
        return JsonResponse(Images.__getImageUrls(self))
