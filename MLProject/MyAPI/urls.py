from django.urls import path, include
from . import views, AlphanumericRecognitionView
from . import flowersViews
from rest_framework import routers
from django.conf import settings
from django.conf.urls.static import static

router = routers.DefaultRouter()
router.register('MyAPI', views.ApprovalsView)
urlpatterns = [
                  path('api/', include(router.urls)),
                  # path('status/', views.approvereject),
                  path('cxform', views.cxcontact, name='cxform'),
                  path('test', views.test, name='test'),
                  path('', views.Home, name='home'),
                  path('flowers/', flowersViews.flowers, name='flowers'),
                  # path('form/', views.cxcontact2, name='cxform2'),
                  path('AlphanumericRecognition/', AlphanumericRecognitionView.AlphanumericRecognition, name='AlphanumericRecognition'),

              ] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
