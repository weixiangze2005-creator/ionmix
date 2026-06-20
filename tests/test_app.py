import os

os.environ["IONMIX_CONDUCTIVITY_MODEL"] = "enabled"

from fastapi.testclient import TestClient

from app.main import app
from app.recommender import canonical_salt


client = TestClient(app)


def test_health_and_model():
    assert client.get("/api/health").json() == {"status": "ok"}
    info = client.get("/api/model-info").json()
    assert info["available"] is True
    assert info["metrics"]["r2"] > 0.5
    assert "LiPF6" in info["supported_salts"]
    assert info["lino3_solubility_model"]["available"] is True
    assert info["lino3_solubility_model"]["metrics"]["rows"] == 25


def test_weight_ui_explains_and_starts_at_one_hundred_percent():
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "总权重 100%" in html
    assert "总和始终为 100%" in html
    assert "上方已选权重保持不变" in html
    assert sum([35, 30, 15, 12, 8]) == 100


def test_salt_alias():
    assert canonical_salt("硝酸锂") == "LiNO3"
    assert canonical_salt(" lipf6 ") == "LiPF6"


def test_lino3_extrapolation_recommendations():
    response = client.post(
        "/api/recommend",
        json={
            "salt": "LiNO3",
            "temperature_c": 25,
            "concentration": 1,
            "application": "lithium_metal",
            "top_k": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["is_extrapolation"] is True
    assert data["training_coverage"]["solubility_labels"] is True
    assert data["training_coverage"]["conductivity_labels"] is False
    assert len(data["recommendations"]) == 10
    solvent_codes = {
        code
        for item in data["recommendations"]
        for code in (item["solvent_a"], item["solvent_b"])
    }
    assert "DMSO" in solvent_codes
    assert all(item["ratio_a"] + item["ratio_b"] == 100 for item in data["recommendations"])
    assert all(item["predicted_solubility_mole_fraction"] is not None for item in data["recommendations"])
    assert len({item["confidence"] for item in data["recommendations"]}) >= 4
    scores = [item["score"] for item in data["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_known_salt_uses_ml():
    response = client.post(
        "/api/recommend",
        json={"salt": "LiPF6", "temperature_c": 25, "concentration": 1, "top_k": 3},
    )
    assert response.status_code == 200
    rows = response.json()["recommendations"]
    assert any(row["predicted_conductivity"] is not None for row in rows)
    scores = [row["score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_arbitrary_weights_change_rankings_and_always_return_candidates():
    profiles = {
        "solubility": [1, 0, 0, 0, 0],
        "conductivity": [0, 1, 0, 0, 0],
        "safety": [0, 0, 1, 0, 0],
        "mixed": [0.07, 0.61, 0.03, 0.24, 0.91],
        "all_zero": [0, 0, 0, 0, 0],
    }
    keys = ["solubility", "conductivity", "safety", "stability", "low_temperature"]
    top_pairs = set()
    for values in profiles.values():
        response = client.post(
            "/api/recommend",
            json={
                "salt": "LiNO3",
                "top_k": 5,
                "weights": dict(zip(keys, values)),
            },
        )
        assert response.status_code == 200
        rows = response.json()["recommendations"]
        assert len(rows) == 5
        top_pairs.add((rows[0]["solvent_a"], rows[0]["solvent_b"]))
    assert len(top_pairs) >= 3


def test_impossible_constraints_return_relaxed_low_confidence_options():
    response = client.post(
        "/api/recommend",
        json={
            "salt": "LiNO3",
            "top_k": 5,
            "min_flash_point_c": 290,
            "max_mixture_viscosity": 0.1,
            "allow_relaxed_fallback": True,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["recommendations"]) == 5
    assert data["search_space"]["used_relaxed_fallback"] is True
    assert all(row["constraint_status"] == "relaxed" for row in data["recommendations"])
    assert all(row["constraint_violations"] for row in data["recommendations"])


def test_threshold_mode_can_return_ternary_candidates():
    response = client.post(
        "/api/recommend",
        json={
            "salt": "LiNO3",
            "max_components": 3,
            "return_all_above_threshold": True,
            "score_threshold": 62,
            "max_results": 20,
        },
    )
    assert response.status_code == 200
    data = response.json()
    rows = data["recommendations"]
    assert len(rows) > 10
    assert len(rows) <= 20
    assert all(row["score"] >= 62 for row in rows)
    assert any(row["component_count"] == 3 for row in rows)
    assert all(sum(component["ratio"] for component in row["components"]) == 100 for row in rows)
