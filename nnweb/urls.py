import os

from django.conf.urls import url, include
from django.contrib import admin

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPOSITORY_ROOT = os.path.dirname(BASE_DIR)

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^retina/', include('retina.urls')),
]

STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(REPOSITORY_ROOT, 'static/')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(REPOSITORY_ROOT, 'media/')