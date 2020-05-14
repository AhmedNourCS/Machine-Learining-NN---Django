from django import forms


class AlphanumericRecognitionForm1(forms.Form):
    firstname = forms.CharField(max_length=15, widget=forms.TextInput(attrs={'placeholder': 'Enter Firstname'}))
