# Ignore  the warnings
import pickle
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from django.contrib import messages
from django.shortcuts import render
from keras import backend as K
from keras.applications.vgg19 import VGG19, decode_predictions, preprocess_input
from keras.preprocessing import image
from matplotlib import style
from rest_framework import viewsets

from .flowersForms import flowersForm
from .models import approvals
from .serializers import approvalsSerializers

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
style.use('fivethirtyeight')
sns.set(style='whitegrid', color_codes=True)


class ApprovalsView(viewsets.ModelViewSet):
    queryset = approvals.objects.all()
    serializer_class = approvalsSerializers


def _load_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    # img = np.expand_dims(img, axis=0)
    # img = preprocess_input(img)
    return img


def flowersPredict(imageURL):
    labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
    loaded_model = pickle.load(open('MyAPI/flowers_VGG19_model.pkl', 'rb'))
    # loaded_model = pickle.load(open('MyAPI/Flowers_Recognition_model.pkl', 'rb'))
    # loaded_model = load_model("MyAPI/Flowers_Recognition_model.h5")
    img = cv2.imread(imageURL)
    img = cv2.resize(img, (150, 150))
    img = np.reshape(img, [-1, 150, 150, 3])
    loaded_classes = loaded_model.predict_classes(img)
    print(labels[loaded_classes[0]])
    return labels[loaded_classes[0]]


from django.core.files.storage import FileSystemStorage


def flowers(request):
    if request.method == 'POST':
        form = flowersForm(request.POST, request.FILES)
        if form.is_valid():
            url = form.cleaned_data['url']
            myfile = request.FILES['url']

            # save file
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            uploaded_file_url = fs.url(filename)

            # Predict
            answer = flowersPredict(uploaded_file_url)
            print(answer)
            # str = settings.MEDIA_ROOT + uploaded_file_url

            nPath = CAM(uploaded_file_url)
            messages.success(request, 'Image Class : {}'.format(answer))
            messages.warning(request, uploaded_file_url)
            messages.info(request,  nPath)


    form = flowersForm()
    return render(request, 'flowers/flowersVGG19.html', {'form': form})


def CAM(imgPath):
    img_path = imgPath  # '../input/flowers-recognition/flowers/sunflower/151898652_b5f1c70b98_n.jpg'
    import os.path
    # print("os.path.exists ",os.path.exists(imgPath))
    org_img = cv2.imread(img_path)

    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x)

    model = VGG19(weights='imagenet')
    preds = model.predict(x)
    cam_predictions = pd.DataFrame(decode_predictions(preds, top=3)[0], columns=['col1', 'category', 'probability']).iloc[:, 1:]

    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]

    last_conv_layer = model.get_layer('block5_conv4')
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(512):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    img = cv2.imread(img_path)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    hif = .8
    superimposed_img = heatmap * hif + img

    import matplotlib.image as mpimg

    output = 'media/cam_output.jpeg'
    cv2.imwrite(output, superimposed_img)
    img = mpimg.imread(output)

    # plt.imshow(img)

    return output
