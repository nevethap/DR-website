from django.conf.urls import url
from django.contrib import admin

from . import views

app_name = 'retina'

urlpatterns = [
    url(r'^$', views.IndexView.as_view(), name='index'),
    url(r'^list/$', views.ListView.as_view(), name='list'),
    url(r'^(?P<pk>[0-9]+)/$', views.DetailView.as_view(), name='detail')
]
