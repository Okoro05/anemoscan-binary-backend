import pytest

from conftest import data_url, make_image_bytes, payload


def create(client, **kwargs):
    response = client.post("/api/captures", json=payload(**kwargs))
    assert response.status_code == 201, response.text
    return response.json()


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["database"] is True
    assert body["modelPresent"] is False
    assert body["modelType"] == "binary-anemia-classifier"
    assert body["classificationThreshold"] == 0.5


def test_create_sets_patient_session_and_returns_scan(client):
    created = create(client)
    assert created["name"] == "patient-001-scan-1"
    assert created["scanNumber"] == 1
    assert created["imageUrl"] == f"/api/captures/{created['id']}/image"
    assert created["anemiaDetected"] is None
    assert created["anemiaProbability"] is None
    assert client.cookies.get("anemoscan_patient_session")


def test_patient_can_only_access_current_scan(client):
    first = create(client)
    assert client.get(first["imageUrl"]).status_code == 200

    second = create(client)
    assert second["scanNumber"] == 2
    assert client.get(second["imageUrl"]).status_code == 200
    assert client.get(first["imageUrl"]).status_code == 403


def test_fresh_browser_cannot_access_another_patients_image(client):
    first = create(client, patient="patient-001")
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as other_client:
        assert other_client.get(first["imageUrl"]).status_code == 401


def test_patient_library_and_dataset_routes_are_not_public(client):
    create(client)
    assert client.get("/api/captures").status_code == 404
    assert client.get("/api/export").status_code == 404
    assert client.patch("/api/captures/1", json={"name": "renamed"}).status_code == 404
    assert client.delete("/api/captures/1").status_code == 404


@pytest.mark.parametrize("bad", ["patient 001", "", "-x", "x--y", "a_b", "x" * 41])
def test_bad_patient_ids_rejected(client, bad):
    assert client.post("/api/captures", json=payload(patient=bad)).status_code == 422


@pytest.mark.parametrize("field,value", [("age", 150), ("age", -1), ("gender", "other"), ("imageData", "nonsense")])
def test_bad_fields_rejected(client, field, value):
    assert client.post("/api/captures", json=payload(**{field: value})).status_code == 422


def test_non_image_rejected(client):
    import base64
    fake = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
    assert client.post("/api/captures", json=payload(imageData=fake)).status_code == 422


def test_gender_shorthand_and_optional_details(client):
    assert create(client, gender="M")["gender"] == "male"
    bare = create(client, patient="patient-009", age=None, gender=None)
    assert bare["age"] is None and bare["gender"] is None


def test_image_roundtrip_and_private_cache(client):
    created = create(client)
    response = client.get(created["imageUrl"])
    assert response.status_code == 200
    assert response.content == make_image_bytes("PNG")
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, no-store"
    assert client.get("/api/captures/9999/image").status_code in (401, 403)


def test_jpeg_and_webp_accepted(client):
    for fmt in ("JPEG", "WEBP"):
        response = client.post("/api/captures", json=payload(patient=f"p-{fmt.lower()}", imageData=data_url(fmt)))
        assert response.status_code == 201


def test_analyze_binary_model(client, use_model):
    use_model(True)
    created = create(client)
    body = client.post(f"/api/captures/{created['id']}/analyze").json()
    assert body["anemiaDetected"] is True
    assert body["anemiaProbability"] > 0.5
    assert body["anemiaStatus"] == "Anemia Detected"
    assert body["analyzedAt"]


def test_analyze_without_patient_session(client, use_model):
    use_model(True)
    created = create(client)
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as other_client:
        assert other_client.post(f"/api/captures/{created['id']}/analyze").status_code == 401


def test_analyze_without_model(client, use_model):
    use_model(False)
    created = create(client)
    assert client.post(f"/api/captures/{created['id']}/analyze").status_code == 503


def test_reanalysing_overwrites_binary_result(client, use_model):
    use_model(True)
    created = create(client)
    first = client.post(f"/api/captures/{created['id']}/analyze").json()
    second = client.post(f"/api/captures/{created['id']}/analyze").json()
    assert second["anemiaProbability"] == first["anemiaProbability"]


def test_cors_preflight(client):
    response = client.options(
        "/api/captures",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", "https://example.com")
