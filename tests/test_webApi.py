
import json
import sys
import os
from app import create_app
import math
app = create_app('test')


def test_health():
    with app.test_client() as c:
        response = c.get('/health')
        assert response.status_code == 200


def test_NonPrank():
    with app.test_client() as c:
        with open(os.getcwd()+'/unit_test/p_data.json') as f:
            json_data = json.load(f)
            response = c.post("/nonlinear/personal_rank", json=json_data)
            #assert response.status_code == 200
            print(response.json)
            for r in response.json['feedItemsRank']:
                g = r.get('global')
                p = r.get('personalised')
                assert type(g) is float
                assert not math.isnan(g)
                assert type(p) is float
                assert not math.isnan(p)

def test_NonGrank():
    with app.test_client() as c:
        with open(os.getcwd()+'/unit_test/g_data.json') as f:
            json_data = json.load(f)
            response = c.post("/nonlinear/global_rank", json=json_data)
            assert response.status_code == 200
            print(response.json)
            for r in response.json['feedItemsRank']:
                g = r.get('global')
                assert type(g) is float
                assert not math.isnan(g)
