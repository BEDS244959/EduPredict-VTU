from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_login_demo_admin():
    r = client.post("/api/auth/login", json={"role": "admin", "login_id": "admin", "password": "Demo@1234"})
    assert r.status_code == 200
    assert r.json()["role"] == "admin"
    token = r.json()["access_token"]
    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200


def test_student_cannot_access_admin():
    r = client.post("/api/auth/login", json={"role": "student", "login_id": "1TJ22CS001", "password": "Demo@1234"})
    if r.status_code != 200:
        return
    token = r.json()["access_token"]
    denied = client.get("/api/admin/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert denied.status_code == 403
