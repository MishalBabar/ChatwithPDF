import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_ask_empty():
    res = client.post('/ask', json={'question': ''})
    assert res.status_code == 400


def test_clear():
    res = client.post('/clear')
    assert res.status_code == 200
    assert res.json() == {'status': 'memory cleared'}


# Mock orchestrator for deterministic test
def test_ask_valid(monkeypatch):
    from orchestrator import workflow

    def fake_handle(self, q):
        return 'test answer'

    monkeypatch.setattr(workflow.Orchestrator, 'handle', fake_handle, raising=True)

    res = client.post('/ask', json={'question': 'Hi'})
    assert res.status_code == 200
    assert res.json() == {'answer': 'test answer'}
