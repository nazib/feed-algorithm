import os
import tempfile
import sys
import pytest
import flask
import glob
from flask import jsonify, make_response
import json
from webApi import app
import pdb
import requests
#@pytest.fixture
def test_health():
    with app.test_client() as c:
        response = c.get('/health')
        json_data = response.get_json()
        assert response.status_code == 200

def test_TrainedModelExists():
     cwd = os.getcwd()
     dirs = os.listdir(cwd)
     model_dir = [x for x in dirs if x=='logs']
     #cwd = os.getcwd().replace('tests','')
     model_path = os.path.join(cwd,model_dir[0])
     models = [glob.glob(os.path.join(model_path,x,'*.h5')).pop() for x in os.listdir(model_path)]
     exists = [os.path.isfile(x) for x in models]
     true_list = [True for x in exists]
     assert exists == true_list
     assert len(exists) !=0


def test_NonPrank():
    with app.test_client() as c:
        with open('unit_test/p_data.json') as f:
            json_data = json.load(f)
        #pdb.set_trace()
        response = requests.post("http://0.0.0.0:5000//nonlinear/personal_rank", json =json_data)
        #response = c.get('/nonlinear/personal_rank', query_string=json.dumps(json_data))
        assert response.status_code == 200

def test_NonGrank():
    with app.test_client() as c:
        with open('unit_test/g_data.json') as f:
            json_data = json.load(f)
        #pdb.set_trace()
        response = requests.post("http://0.0.0.0:5000//nonlinear/global_rank", json =json_data)
        #response = c.get('/nonlinear/personal_rank', query_string=json.dumps(json_data))
        assert response.status_code == 200

def test_LinGrank():
    with app.test_client() as c:
        with open('unit_test/g_data.json') as f:
            json_data = json.load(f)
        #pdb.set_trace()
        response = requests.post("http://0.0.0.0:5000//linear/global_rank", json =json_data)
        #response = c.get('/nonlinear/personal_rank', query_string=json.dumps(json_data))
        assert response.status_code == 200

def test_LinPrank():
    with app.test_client() as c:
        with open('unit_test/p_data.json') as f:
            json_data = json.load(f)
        response = requests.post("http://0.0.0.0:5000//linear/personal_rank", json =json_data)
        #response = c.get('/nonlinear/personal_rank', query_string=json.dumps(json_data))
        assert response.status_code == 200

'''
if __name__=="__main__":
    test_NonPrank()
'''