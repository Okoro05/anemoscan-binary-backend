import csv
import io
import zipfile

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


def test_names_and_scan_numbers(client):
    first = create(client)
    second = create(client)
    other = create(client, patient="patient-002")
    assert first["name"] == "patient-001-scan-1" and first["scanNumber"] == 1
    assert second["name"] == "patient-001-scan-2" and second["scanNumber"] == 2
    assert other["name"] == "patient-002-scan-1" and other["scanNumber"] == 1
    assert first["age"] == 25 and first["gender"] == "female"
    assert first["imageUrl"] == f"/api/captures/{first['id']}/image"
    assert first["anemiaDetected"] is None
    assert first["anemiaProbability"] is None


def test_patient_id_is_normalised(client):
    assert create(client, patient="Patient-007")["patientId"] == "patient-007"


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


def test_list_order_and_pagination(client):
    for _ in range(10):
        create(client)
    page1 = client.get("/api/captures").json()
    assert len(page1["captures"]) == 8 and page1["hasMore"] is True
    assert page1["captures"][0]["scanNumber"] == 10
    page2 = client.get("/api/captures?offset=8").json()
    assert len(page2["captures"]) == 2 and page2["hasMore"] is False
    assert client.get("/api/captures?limit=100").status_code == 422


def test_image_roundtrip(client):
    created = create(client)
    response = client.get(created["imageUrl"])
    assert response.status_code == 200
    assert response.content == make_image_bytes("PNG")
    assert response.headers["content-type"] == "image/png"
    assert "immutable" in response.headers["cache-control"]
    assert client.get("/api/captures/9999/image").status_code == 404


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
    assert client.get("/api/captures").json()["captures"][0]["anemiaDetected"] is True


def test_analyze_without_model(client, use_model):
    use_model(False)
    created = create(client)
    assert client.post(f"/api/captures/{created['id']}/analyze").status_code == 503
    assert client.post("/api/captures/9999/analyze").status_code == 404


def test_reanalysing_overwrites_binary_result(client, use_model):
    use_model(True)
    created = create(client)
    first = client.post(f"/api/captures/{created['id']}/analyze").json()
    second = client.post(f"/api/captures/{created['id']}/analyze").json()
    assert second["anemiaProbability"] == first["anemiaProbability"]


def test_export_zip(client):
    create(client)
    create(client, patient="patient-002", age=61, gender="male")
    response = client.get("/api/export")
    assert response.status_code == 200
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    names = archive.namelist()
    assert "images/patient-001-scan-1.png" in names and "images/patient-002-scan-1.png" in names
    rows = list(csv.DictReader(io.StringIO(archive.read("anemia_scans.csv").decode())))
    assert [row["image"] for row in rows] == ["patient-001-scan-1.png", "patient-002-scan-1.png"]
    assert rows[0]["anemia_detected"] == "" and rows[1]["gender"] == "male"


def test_cors_preflight(client):
    response = client.options(
        "/api/captures",
        headers={"Origin": "https://example.com", "Access-Control-Request-Method": "POST"},
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] in ("*", "https://example.com")
