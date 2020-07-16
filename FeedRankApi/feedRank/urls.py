from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns
from feedRank import views

urlpatterns = [
    path('feedRank/', views.rank),
    path('feedRank/bulk', views.bulk_rank),
]
