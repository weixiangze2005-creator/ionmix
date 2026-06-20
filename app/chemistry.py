from __future__ import annotations

from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


DESCRIPTOR_NAMES = (
    "molecular_weight",
    "tpsa",
    "logp",
    "hbond_acceptors",
    "hbond_donors",
    "rotatable_bonds",
    "fraction_csp3",
    "ring_count",
)


@lru_cache(maxsize=512)
def molecular_descriptors(smiles: str) -> dict[str, float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "hbond_acceptors": float(Lipinski.NumHAcceptors(mol)),
        "hbond_donors": float(Lipinski.NumHDonors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "ring_count": float(Lipinski.RingCount(mol)),
    }


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)

