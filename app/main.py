from __future__ import annotations

import logging
import json
import re
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from app import auth_store
from app.compatibility import public_rule_catalog
from app.recommender import FormulationRecommender, RecommendationOptions
from app.schemas import AuthRequest, RecommendationRequest, SavedFormulaRequest


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "app" / "static"
logger = logging.getLogger("ionmix.account_storage")

app = FastAPI(
    title="Electrolyte Formulation Explorer",
    version="0.1.0",
    description="公开数据驱动的电解液二元溶剂与配比预筛选系统",
)
app.add_middleware(GZipMiddleware, minimum_size=900)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

recommender = FormulationRecommender()
_account_status_cache: tuple[float, dict] | None = None
_account_status_lock = Lock()
_recommend_cache: OrderedDict[str, dict] = OrderedDict()
_recommend_cache_lock = Lock()
_RECOMMEND_CACHE_SIZE = 12


def safe_database_error(exc: Exception) -> str:
    message = str(exc)
    database_url = auth_store.database_url()
    if database_url:
        message = message.replace(database_url, "[database-url-redacted]")
    message = re.sub(
        r"(postgres(?:ql)?://)([^:@\s]+)(?::([^@\s]*))?@",
        r"\1[redacted]@",
        message,
        flags=re.IGNORECASE,
    )
    return f"{exc.__class__.__name__}: {message}"[:600]


def account_storage_status() -> dict:
    global _account_status_cache
    now = monotonic()
    with _account_status_lock:
        if _account_status_cache and now - _account_status_cache[0] < 30.0:
            return dict(_account_status_cache[1])
    backend = "postgresql" if auth_store.using_postgres() else "sqlite"
    try:
        auth_store.init_db()
    except Exception as exc:
        logger.warning("Account database unavailable: %s", safe_database_error(exc))
        status = {
            "backend": backend,
            "persistent": False,
            "available": False,
        }
    else:
        status = {
            "backend": backend,
            "persistent": bool(auth_store.using_postgres()),
            "available": True,
        }
    with _account_status_lock:
        _account_status_cache = (now, dict(status))
    return status


def ensure_account_storage() -> None:
    try:
        auth_store.init_db()
    except Exception as exc:
        logger.warning("Account database unavailable: %s", safe_database_error(exc))
        raise HTTPException(
            status_code=503,
            detail="账户存储暂时不可用，请检查 Render 的 Supabase 数据库连接配置。",
        ) from exc


def current_user_optional(request: Request) -> dict | None:
    token = request.cookies.get(auth_store.SESSION_COOKIE)
    if not token:
        return None
    try:
        return auth_store.user_from_session(token)
    except Exception:
        return None


def current_user(request: Request) -> dict:
    token = request.cookies.get(auth_store.SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=401, detail="请先登录。")
    try:
        user = auth_store.user_from_session(token)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="账户存储暂时不可用，请稍后重试。",
        ) from exc
    if not user:
        raise HTTPException(status_code=401, detail="请先登录。")
    return user


def set_session_cookie(response: Response, token: str, expires_at: str) -> None:
    response.set_cookie(
        auth_store.SESSION_COOKIE,
        token,
        httponly=True,
        samesite="lax",
        max_age=auth_store.SESSION_DAYS * 24 * 60 * 60,
        expires=expires_at,
        path="/",
    )


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/auth/register")
def register(payload: AuthRequest, response: Response):
    ensure_account_storage()
    try:
        user = auth_store.create_user(
            payload.email,
            payload.password,
            payload.display_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session = auth_store.create_session(user["id"])
    set_session_cookie(response, session["token"], session["expires_at"])
    return {"user": user}


@app.post("/api/auth/login")
def login(payload: AuthRequest, response: Response):
    ensure_account_storage()
    user = auth_store.authenticate_user(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail="邮箱或密码不正确。")
    session = auth_store.create_session(user["id"])
    set_session_cookie(response, session["token"], session["expires_at"])
    return {"user": user}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    ensure_account_storage()
    auth_store.delete_session(request.cookies.get(auth_store.SESSION_COOKIE))
    response.delete_cookie(auth_store.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/api/auth/me")
def me(request: Request):
    return {"user": current_user_optional(request)}


@app.get("/api/history")
def history(user: dict = Depends(current_user)):
    return {"items": auth_store.list_formulations(user["id"])}


@app.post("/api/history")
def save_history(payload: SavedFormulaRequest, user: dict = Depends(current_user)):
    try:
        item = auth_store.save_formulation(
            user_id=user["id"],
            name=payload.name,
            recommendation=payload.recommendation,
            request_context=payload.request_context,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"item": item}


@app.delete("/api/history/{formulation_id}")
def delete_history(formulation_id: int, user: dict = Depends(current_user)):
    deleted = auth_store.delete_formulation(user["id"], formulation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="没有找到这条历史配方。")
    return {"ok": True}


@app.get("/api/solvents")
def solvents():
    df = recommender.catalog
    columns = [
        "code", "name", "smiles", "dielectric_constant", "viscosity_mpas",
        "flash_point_c", "donor_number", "battery_role", "pubchem_url",
    ]
    return df[columns].to_dict(orient="records")


@app.get("/api/model-info")
def model_info():
    model = recommender.model
    lino3_model = recommender.lino3_model
    mixture_model = recommender.mixture_model
    oedb_model = recommender.oedb_model
    catalog = recommender.catalog
    compatibility_rules = public_rule_catalog()
    return {
        "available": model.available,
        "metrics": model.metrics,
        "supported_salts": model.supported_salts,
        "supported_solvents": model.supported_solvents,
        "training_dataset": "CALiSol-23",
        "dataset_url": "https://github.com/Pele0599/CALiSol-23",
        "solvent_catalog": {
            "count": int(len(catalog)),
            "oedb_overlap_count": int(catalog["oedb_code"].astype(str).str.len().gt(0).sum()),
        },
        "compatibility_rules": {
            "enabled": True,
            "count": len(compatibility_rules),
            "rules": compatibility_rules,
        },
        "account_storage": {
            **account_storage_status(),
        },
        "lino3_solubility_model": {
            "available": lino3_model.available,
            "metrics": lino3_model.metrics,
            "source_doi": "10.1016/j.fluid.2017.12.034",
        },
        "mixture_property_model": {
            "available": mixture_model.available,
            "metrics": mixture_model.metrics,
            "source_csv": "data/mixture_experiments.csv",
        },
        "oedb_auxiliary_model": {
            "available": oedb_model.available,
            "metrics": oedb_model.metrics,
            "supported_salts": oedb_model.supported_salts,
            "supported_solvents": oedb_model.supported_solvents,
            "data_type": "OEDB high-throughput molecular dynamics simulation",
        },
    }


@app.post("/api/recommend")
def recommend(request: RecommendationRequest, response: Response):
    started = perf_counter()
    cache_key = json.dumps(request.model_dump(), sort_keys=True, ensure_ascii=False)
    with _recommend_cache_lock:
        cached = _recommend_cache.get(cache_key)
        if cached is not None:
            _recommend_cache.move_to_end(cache_key)
    if cached is not None:
        result = deepcopy(cached)
        result["runtime"] = {
            "elapsed_ms": round((perf_counter() - started) * 1000.0, 1),
            "cache_hit": True,
        }
        response.headers["X-Ionmix-Cache"] = "HIT"
        return result

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
        score_threshold=request.score_threshold,
        max_results=request.max_results,
        max_components=request.max_components,
        return_all_above_threshold=request.return_all_above_threshold,
        weights=request.weights.model_dump(),
        allow_relaxed_fallback=request.allow_relaxed_fallback,
    )
    result = recommender.recommend(options)
    result["runtime"] = {
        "elapsed_ms": round((perf_counter() - started) * 1000.0, 1),
        "cache_hit": False,
    }
    with _recommend_cache_lock:
        _recommend_cache[cache_key] = deepcopy(result)
        _recommend_cache.move_to_end(cache_key)
        while len(_recommend_cache) > _RECOMMEND_CACHE_SIZE:
            _recommend_cache.popitem(last=False)
    response.headers["X-Ionmix-Cache"] = "MISS"
    return result
