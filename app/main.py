from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.catalog import load_catalog
from app.ml_model import ConductivityModel
from app.lino3_model import LiNO3SolubilityModel
from app.recommender import FormulationRecommender, RecommendationOptions
from app.schemas import RecommendationRequest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"

app = FastAPI(
    title="Electrolyte Formulation Explorer",
    version="0.1.0",
    description="公开数据驱动的电解液二元溶剂与配比预筛选系统",
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

recommender = FormulationRecommender()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/solvents")
def solvents():
    df = load_catalog()
    columns = [
        "code", "name", "smiles", "dielectric_constant", "viscosity_mpas",
        "flash_point_c", "donor_number", "battery_role", "pubchem_url",
    ]
    return df[columns].to_dict(orient="records")


@app.get("/api/model-info")
def model_info():
    model = ConductivityModel()
    lino3_model = LiNO3SolubilityModel()
    return {
        "available": model.available,
        "metrics": model.metrics,
        "supported_salts": model.supported_salts,
        "supported_solvents": model.supported_solvents,
        "training_dataset": "CALiSol-23",
        "dataset_url": "https://github.com/Pele0599/CALiSol-23",
        "lino3_solubility_model": {
            "available": lino3_model.available,
            "metrics": lino3_model.metrics,
            "source_doi": "10.1016/j.fluid.2017.12.034",
        },
    }


@app.post("/api/recommend")
def recommend(request: RecommendationRequest):
    options = RecommendationOptions(
        salt=request.salt,
        temperature_c=request.temperature_c,
        concentration=request.concentration,
        concentration_unit=request.concentration_unit,
        min_flash_point_c=request.min_flash_point_c,
        max_mixture_viscosity=request.max_mixture_viscosity,
        exclude_high_hazard=request.exclude_high_hazard,
        application=request.application,
        top_k=request.top_k,
        weights=request.weights.model_dump(),
        allow_relaxed_fallback=request.allow_relaxed_fallback,
    )
    return recommender.recommend(options)
