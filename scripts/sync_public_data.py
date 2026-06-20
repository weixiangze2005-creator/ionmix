from __future__ import annotations

import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

from app.chemistry import canonicalize_smiles, molecular_descriptors


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CALISOL_URL = "https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/CALiSol-23%20Dataset.csv"
CALISOL_SMILES_URL = "https://raw.githubusercontent.com/Pele0599/CALiSol-23/main/calisolsmile.csv"
PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"


def download(url: str, destination: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def pubchem_record(name: str) -> dict:
    properties = "ConnectivitySMILES,MolecularWeight,XLogP,TPSA,HBondDonorCount,HBondAcceptorCount"
    url = f"{PUBCHEM_BASE}/{quote(name)}/property/{properties}/JSON"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    record = response.json()["PropertyTable"]["Properties"][0]
    cid = record.get("CID")
    return {
        "pubchem_cid": cid,
        "pubchem_url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid else "",
        "pubchem_canonical_smiles": record.get("ConnectivitySMILES") or record.get("CanonicalSMILES"),
        "pubchem_molecular_weight": record.get("MolecularWeight"),
        "pubchem_xlogp": record.get("XLogP"),
        "pubchem_tpsa": record.get("TPSA"),
        "pubchem_hbond_donors": record.get("HBondDonorCount"),
        "pubchem_hbond_acceptors": record.get("HBondAcceptorCount"),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    download(CALISOL_URL, RAW_DIR / "calisol23.csv")
    download(CALISOL_SMILES_URL, RAW_DIR / "calisol_smiles.csv")

    catalog = pd.read_csv(ROOT / "data" / "solvent_seed.csv")
    records = []
    failures = []
    for row in catalog.to_dict(orient="records"):
        row["smiles"] = canonicalize_smiles(row["smiles"])
        row.update(molecular_descriptors(row["smiles"]))
        try:
            row.update(pubchem_record(row["name"]))
        except Exception as exc:
            failures.append({"name": row["name"], "error": str(exc)})
            row.update({"pubchem_cid": None, "pubchem_url": ""})
        records.append(row)
        time.sleep(0.08)

    pd.DataFrame(records).to_csv(PROCESSED_DIR / "solvent_catalog.csv", index=False)
    metadata = {
        "calisol_url": CALISOL_URL,
        "calisol_smiles_url": CALISOL_SMILES_URL,
        "pubchem_api": "PUG REST",
        "pubchem_api_docs": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
        "records": len(records),
        "pubchem_failures": failures,
        "note": "Physical-property seed values are screening priors. PubChem supplies identity and molecular descriptors.",
    }
    (PROCESSED_DIR / "source_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Synced CALiSol-23 and {len(records)} solvent records.")
    if failures:
        print(f"PubChem lookups failed for {len(failures)} records; RDKit descriptors remain available.")


if __name__ == "__main__":
    main()
