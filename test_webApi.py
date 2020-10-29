import os
import glob
import json
from .app import app
import requests


def test_health():
    with app.test_client() as c:
        response = c.get('/health')
        assert response.status_code == 200


def test_NonPrank():
    with app.test_client():
        with open('unit_test/p_data.json') as f:
            json_data = json.load(f)
        response = requests.post(
            "http://0.0.0.0:8080//nonlinear/personal_rank", json=json_data)
        assert response.status_code == 200


def test_NonGrank():
    with app.test_client():
        with open('unit_test/g_data.json') as f:
            json_data = json.load(f)
        # pdb.set_trace()
        response = requests.post(
            "http://0.0.0.0:8080//nonlinear/global_rank", json=json_data)
        assert response.status_code == 200
