
import json
import app
from app import create_app

app = create_app('test')

def test_health():
    with app.test_client() as c:
        response = c.get('/health')
        assert response.status_code == 200

def test_NonPrank():
    with app.test_client() as c:
        with open('unit_test/p_data.json') as f:
            json_data = json.load(f)
        response = c.post(
            "/nonlinear/personal_rank", json=json_data)
        assert response.status_code == 200

def test_NonGrank():
    with app.test_client() as c:
        with open('unit_test/g_data.json') as f:
            json_data = json.load(f)
        # pdb.set_trace()
        response = c.post(
            "/nonlinear/global_rank", json=json_data)
        print(response)
        assert response.status_code == 200
