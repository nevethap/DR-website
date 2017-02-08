from __future__ import unicode_literals

from django.db import models
from django.core.urlresolvers import reverse


class Image(models.Model):
    srcFile = models.CharField(max_length=500)
    actual = models.IntegerField()
    preidcted = models.IntegerField(null=True)
    img_logo = models.CharField(max_length=1000)

    def __str__(self):
        return self.srcFile + '-' + str(self.actual)

    def get_absolute_url(self):
        return reverse('retina:detail', kwargs={'pk': self.pk})
