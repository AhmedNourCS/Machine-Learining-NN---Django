from django import forms
from .flowersModels import *


class flowersForm(forms.ModelForm):
    class Meta:
        model = flowersModel
        fields = ['url']
