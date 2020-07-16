from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

# Create your views here.
def rankapp(request):
    #template = loader.get_template("rankapp/page.html")
    return render(request,"page.html",{})
