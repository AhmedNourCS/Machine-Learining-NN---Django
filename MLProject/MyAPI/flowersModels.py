from django.db import models


# Create your models here.
class flowersModel(models.Model):
    url = models.ImageField(upload_to='images/')
