from django.urls import path
from . import views

urlpatterns = [
	path('', views.homepage, name='home'),
	path('fileupload/', views.fileUpload, name='fileupload'),
	path('ajax/checkprogress/', views.checkProgress, name='checkprogress'),
]