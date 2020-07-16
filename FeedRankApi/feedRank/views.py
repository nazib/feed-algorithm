from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.parsers import JSONParser
from . model import*
from . preprocess_data import*
from . rank_logics import*
from .apps import FeedrankConfig
import pandas as pd
import numpy as np
import json
import pdb 
# Create your views here.

@api_view(['POST'])
def rank(request):
    #data = np.array([12,10,8,3,16,18,20,8],dtype=float)
    data = JSONParser().parse(request)
    data = np.array(list(data.values()),dtype=float)
    TH= 60*36
    values = data[2]
    decay = FeedrankConfig.theta[2]*np.exp(1-(values/TH))
    rank = np.sum(data*FeedrankConfig.theta)
    response = {"RankScore":rank}
    return JsonResponse(response)

@api_view(['POST'])
def bulk_rank(request):
    feed_data = JSONParser().parse(request)
    feed_frame = pd.read_json(feed_data,typ='frame', orient='split')
    feed_frame_num = feed_frame.drop(["uid","ptid","feed_id"],axis=1)
    rank_scores = GlobalRank(feed_frame_num.values,FeedrankConfig.theta)
    feed_frame['RankScore'] = rank_scores
    return JsonResponse(feed_frame.to_json(),safe=False)
