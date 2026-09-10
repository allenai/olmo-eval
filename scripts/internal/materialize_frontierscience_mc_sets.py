"""Materialize and mechanically validate FrontierScience MC choice sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from build_frontierscience_mc_distractors import extract_tagged_values, normalized_text

CHOICE_LABELS = "ABCD"

# Small, transparent display/curation repairs identified by mechanical review.
# A normalized ``reference_answer`` is supplied when the source presentation
# itself would otherwise leak the gold choice.
MANUAL_OVERRIDES: dict[str, dict[str, Any]] = {
    "0ea11f5b-df09-4330-92cc-302a63c22008": {
        "display_gold": (
            r"\(\frac{\Delta T}{\Delta h}=-\frac{Mg\beta}{R\alpha}"
            r"\frac{1+\alpha\rho_s}{1+\alpha\beta\rho_s}\)"
        ),
        "display_distractors": [
            (
                r"\(\frac{\Delta T}{\Delta h}=-\frac{Mg\alpha}{R\beta}"
                r"\frac{1+\alpha\rho_s}{1+\alpha\beta\rho_s}\)"
            ),
            (
                r"\(\frac{\Delta T}{\Delta h}=-\frac{Mg\beta}{R\alpha}"
                r"\frac{1+\alpha\beta\rho_s}{1+\alpha\rho_s}\)"
            ),
            (
                r"\(\frac{\Delta T}{\Delta h}=-\frac{Mg\beta}{R\alpha}"
                r"\frac{1+\beta\rho_s}{1+\alpha\beta\rho_s}\)"
            ),
        ],
        "reference_answer": (
            r"\(\frac{\Delta T}{\Delta h}=-\frac{Mg\beta}{R\alpha}"
            r"\frac{1+\alpha\rho_s}{1+\alpha\beta\rho_s}\)"
        ),
        "curation_tier": "strict_clean",
        "reason": "Normalized all four list choices to the same inline-math convention.",
    },
    "27c865e6-1c87-489b-b7ea-b197fe3356ba": {
        "display_gold": r"\(U=CT\)",
        "display_distractors": [r"\(U=2CT\)", r"\(U=\frac{1}{2}CT\)", r"\(U=CT^2\)"],
        "curation_tier": "curated_exception",
        "reason": (
            "Removed a stray apostrophe and replaced P-dependent choices that violated the "
            "requested C,T-only answer form."
        ),
        "provenance": [
            {
                "source_id": "manual-override-1",
                "answer": r"\(U=2CT\)",
                "error_mechanism": "Double-counts the thermal-energy contribution.",
            },
            {
                "source_id": "manual-override-2",
                "answer": r"\(U=\frac{1}{2}CT\)",
                "error_mechanism": "Introduces an erroneous one-half equipartition factor.",
            },
            {
                "source_id": "manual-override-3",
                "answer": r"\(U=CT^2\)",
                "error_mechanism": "Uses an incorrect quadratic temperature dependence.",
            },
        ],
    },
    "422512ca-5a0c-4a23-b987-7fa2eb2024e6": {
        "display_gold": "19",
        "display_distractors": ["20", "10", "38"],
        "curation_tier": "strict_clean",
        "reason": "Normalized the gold from r=19 to a bare number to match all distractors.",
    },
    "f398c3ad-6cff-45b1-a21d-5ce963db75aa": {
        "display_gold": r"\(\left(-R\left[1+\frac{5}{12}\alpha\right],0\right)\)",
        "display_distractors": [
            r"\(\left(-R\left[1-\frac{5}{12}\alpha\right],0\right)\)",
            r"\(\left(-R\left[1+\frac{1}{12}\alpha\right],0\right)\)",
            r"\(\left(-R\left[1+\frac{5}{12}\beta\right],0\right)\)",
        ],
        "reference_answer": r"\(\left(-R\left[1+\frac{5}{12}\alpha\right],0\right)\)",
        "curation_tier": "strict_clean",
        "reason": (
            "Normalized all four choices to the same delimiters and first-power notation; "
            "alpha^1 is exactly alpha."
        ),
    },
    "dbe975d5-c431-4d73-8f7c-f8bba77ea458": {
        "display_gold": (
            r"\(\left|\vec F_Y\right|="
            r"\frac{8\pi\epsilon\sigma T_0^3T_1}{3c}R^2\)"
        ),
        "display_distractors": [
            (
                r"\(\left|\vec F_Y\right|="
                r"\frac{4\pi\epsilon\sigma T_0^3T_1}{c}R^2\)"
            ),
            (
                r"\(\left|\vec F_Y\right|="
                r"\frac{4\pi\epsilon\sigma T_0^3T_1}{3c}R^2\)"
            ),
            (
                r"\(\left|\vec F_Y\right|="
                r"\frac{8\pi\epsilon\sigma T_0^3T_1}{3\sqrt{2}\,c}R^2\)"
            ),
        ],
        "reference_answer": (
            r"\(\left|\vec F_Y\right|="
            r"\frac{8\pi\epsilon\sigma T_0^3T_1}{3c}R^2\)"
        ),
        "curation_tier": "strict_clean",
        "reason": (
            "Matched the requested force magnitude notation across choices and removed "
            "source-only punctuation."
        ),
    },
    "16318af5-cfa6-4b2c-bf25-9eb3ac468b48": {
        "display_gold": r"\(-\frac{f}{2T}\)",
        "display_distractors": [
            r"\(\frac{f}{2T}\)",
            r"\(-\frac{f}{T}\)",
            r"\(-\frac{3f}{2T}\)",
        ],
        "reference_answer": r"\(-\frac{f}{2T}\)",
        "curation_tier": "strict_clean",
        "reason": "Normalized whitespace and math delimiters across all four choices.",
    },
    "20fcf049-b356-45cf-83d8-833f280ee107": {
        "display_gold": r"\(\alpha=\frac{J}{k_BT}\)",
        "display_distractors": [
            r"\(\alpha=\frac{J}{2k_BT}\)",
            r"\(\alpha=\frac{2J}{k_BT}\)",
            r"\(\alpha=\frac{J}{3k_BT}\)",
        ],
        "reference_answer": r"\(\alpha=\frac{J}{k_BT}\)",
        "curation_tier": "strict_clean",
        "reason": "Used the same equation and fraction convention for every choice.",
    },
    "f88cc0c0-2ca6-4cb0-bd1e-a37ddcf11294": {
        "display_gold": r"\(E=-8.6\times10^{39}\,\mathrm{J}\)",
        "display_distractors": [
            r"\(E=-1.6\times10^{40}\,\mathrm{J}\)",
            r"\(E=-5.4\times10^{39}\,\mathrm{J}\)",
            r"\(E=-1.1\times10^{39}\,\mathrm{J}\)",
        ],
        "reference_answer": r"\(E=-8.6\times10^{39}\,\mathrm{J}\)",
        "curation_tier": "strict_clean",
        "reason": "Braced all two-digit exponents and normalized scientific notation.",
    },
    "218df0c2-18db-495f-8c53-eb6758feaff2": {
        "display_gold": (
            r"\(\frac{\tau_f}{\tau_i}="
            r"\sqrt{\frac{b+aH_f^2}{b+aH_i^2}}\)"
        ),
        "display_distractors": [
            (
                r"\(\frac{\tau_f}{\tau_i}="
                r"\sqrt{\frac{b+aH_i^2}{b+aH_f^2}}\)"
            ),
            r"\(\frac{\tau_f}{\tau_i}=\frac{b+aH_f^2}{b+aH_i^2}\)",
            (
                r"\(\frac{\tau_f}{\tau_i}="
                r"\exp\!\left[\frac{a(H_f^2-H_i^2)}{2b}\right]\)"
            ),
        ],
        "reference_answer": (
            r"\(\frac{\tau_f}{\tau_i}="
            r"\sqrt{\frac{b+aH_f^2}{b+aH_i^2}}\)"
        ),
        "curation_tier": "strict_clean",
        "reason": "Included the requested temperature ratio on every choice.",
    },
    "19962a30-eaf9-4b7e-86c7-17c02cdb01bd": {
        "display_gold": (
            r"\(\sigma(x,y,z)=\frac{Q}{4\pi abc"
            r"\sqrt{\frac{x^2}{a^4}+\frac{y^2}{b^4}+\frac{z^2}{c^4}}}\)"
        ),
        "display_distractors": [
            (
                r"\(\sigma(x,y,z)=\frac{Q}{4\pi abc"
                r"\sqrt{\frac{x^2}{a^2b^2}+\frac{y^2}{b^2c^2}"
                r"+\frac{z^2}{c^2a^2}}}\)"
            ),
            (
                r"\(\sigma(x,y,z)=\frac{Q}{4\pi(abc)^{4/3}"
                r"\sqrt{\frac{x^2}{a^6}+\frac{y^2}{b^6}+\frac{z^2}{c^6}}}\)"
            ),
            (
                r"\(\sigma(x,y,z)=\frac{Q}{4\pi(abc)^{1/3}}"
                r"\sqrt{\frac{x^2}{a^4}+\frac{y^2}{b^4}+\frac{z^2}{c^4}}\)"
            ),
        ],
        "reference_answer": (
            r"\(\sigma(x,y,z)=\frac{Q}{4\pi abc"
            r"\sqrt{\frac{x^2}{a^4}+\frac{y^2}{b^4}+\frac{z^2}{c^4}}}\)"
        ),
        "curation_tier": "strict_clean",
        "reason": "Used the stem's sigma(x,y,z) notation for every choice.",
    },
    "eec8840a-2d00-4e70-b043-0da51bd1b288": {
        "display_gold": "Sulfuryl chloride (SO2Cl2)",
        "display_distractors": [
            "Thionyl chloride (SOCl2)",
            "Chlorosulfonic acid (HSO3Cl)",
            "Pyrosulfuryl chloride (S2O5Cl2)",
        ],
        "reference_answer": "Sulfuryl chloride (SO2Cl2)",
        "curation_tier": "strict_clean",
        "reason": "Used the same plain chemical-formula convention for every choice.",
    },
    "6681ea07-9b3a-4a71-9b31-a88212189f27": {
        "display_gold": r"\(2.10\times10^{-6}\,\mathrm{M}\)",
        "display_distractors": [
            r"\(5.30\times10^{-7}\,\mathrm{M}\)",
            r"\(2.03\times10^{-6}\,\mathrm{M}\)",
            r"\(2.73\times10^{-6}\,\mathrm{M}\)",
        ],
        "reference_answer": r"\(2.10\times10^{-6}\,\mathrm{M}\)",
        "curation_tier": "strict_clean",
        "reason": "Removed the gold-only prose stem and normalized all numeric quantities.",
    },
    "f12a8706-3ddd-4255-9234-0655babceb8e": {
        "display_gold": (
            r"\(S_2O_3^{2-}\) will disproportionate, with "
            r"\(E=0.08\,\mathrm{V}\)."
        ),
        "display_distractors": [
            (
                r"\(S_2O_3^{2-}\) will form by comproportionation, with "
                r"\(E=0.08\,\mathrm{V}\)."
            ),
            (
                r"\(S_2O_3^{2-}\) will disproportionate, with "
                r"\(E=0.47\,\mathrm{V}\)."
            ),
            (
                r"\(S_2O_3^{2-}\) will disproportionate, with "
                r"\(E=0.21\,\mathrm{V}\)."
            ),
        ],
        "reference_answer": (
            r"\(S_2O_3^{2-}\) will disproportionate, with "
            r"\(E=0.08\,\mathrm{V}\)."
        ),
        "curation_tier": "strict_clean",
        "reason": "Removed the remaining unmatched Markdown delimiter and normalized units.",
    },
    "6e9aca19-46c4-46ea-9643-86b138e8f2f5": {
        "display_gold": r"\(\vec{PA}\)",
        "display_distractors": [r"\(\vec{PT}\)", r"\(\vec{PP}\)", r"\(\vec{PC}\)"],
        "curation_tier": "curated_exception",
        "usable": True,
        "reason": (
            "Used the complete remaining answer space because the stem explicitly restricts x "
            "to A, T, P, or C."
        ),
        "provenance": [
            {
                "source_id": "constrained-space-1",
                "answer": r"\(\vec{PT}\)",
                "error_mechanism": "Uses the impact point rather than the aim point.",
            },
            {
                "source_id": "constrained-space-2",
                "answer": r"\(\vec{PP}\)",
                "error_mechanism": "Treats the final direction as the zero displacement at P.",
            },
            {
                "source_id": "constrained-space-3",
                "answer": r"\(\vec{PC}\)",
                "error_mechanism": "Uses the vertical radius rather than planar velocity.",
            },
        ],
    },
    "94c55a69-1e8d-4934-ae59-2cf0401e1b01": {
        "display_gold": (
            r"\(y(x)=-\frac{\rho g}{6Yr^2}x^2(x^2-4Lx+6L^2),\quad "
            r"\alpha=-\frac{2\rho gL^3}{3Yr^2}\)"
        ),
        "display_distractors": [
            (
                r"\(y(x)=-\frac{\rho g}{24Yr^2}x^2(x^2-4Lx+6L^2),\quad "
                r"\alpha=-\frac{\rho gL^3}{6Yr^2}\)"
            ),
            (
                r"\(y(x)=-\frac{2\rho gL}{3Yr^2}x^2(3L-x),\quad "
                r"\alpha=-\frac{2\rho gL^3}{Yr^2}\)"
            ),
            (
                r"\(y(x)=-\frac{\rho g}{6Yr^2}x^2(x^2-3Lx+6L^2),\quad "
                r"\alpha=-\frac{7\rho gL^3}{6Yr^2}\)"
            ),
        ],
        "reference_answer": (
            r"\(y(x)=-\frac{\rho g}{6Yr^2}x^2(x^2-4Lx+6L^2),\quad "
            r"\alpha=-\frac{2\rho gL^3}{3Yr^2}\)"
        ),
        "curation_tier": "strict_clean",
        "reason": "Joined the two gold components to match the other three choices.",
    },
    "4195adea-9fa5-4f25-91ef-ebd6c57cfb88": {
        "display_gold": r"\(T=T_0\left(\frac{N}{N_0}\right)^{\frac{1}{5}}\)",
        "display_distractors": [
            r"\(T=T_0\left(\frac{N}{N_0}\right)^{\frac{1}{3}}\)",
            r"\(T=T_0\left(\frac{N}{N_0}\right)^{\frac{2}{5}}\)",
            r"\(T=T_0\left(\frac{N}{N_0}\right)^{\frac{1}{7}}\)",
        ],
        "reference_answer": r"\(T=T_0\left(\frac{N}{N_0}\right)^{\frac{1}{5}}\)",
        "curation_tier": "strict_clean",
        "reason": "Used stacked fractions for every fractional exponent.",
    },
    "cfa05761-b488-4491-9f77-8854ece1cf98": {
        "display_gold": (
            r"\(z_i=(d+D)\theta+\frac{w\theta}{1+\xi C_0"
            r"\exp\!\left(-\frac{4\pi(\rho-\rho_0)a^3g}{3k_BT}d\theta\right)}\)"
        ),
        "display_distractors": [
            (
                r"\(z_i=(d+D)\theta+\frac{w\theta}{1+\xi C_0"
                r"\exp\!\left(-\frac{4\pi\rho a^3g}{3k_BT}d\theta\right)}\)"
            ),
            (
                r"\(z_i=d\theta+\frac{(w+D)\theta}{1+\xi C_0"
                r"\exp\!\left(-\frac{4\pi(\rho-\rho_0)a^3g}{3k_BT}d\theta\right)}\)"
            ),
            (
                r"\(z_i=(d+D)\theta+w\theta\left[1+\xi C_0"
                r"\exp\!\left(-\frac{4\pi(\rho-\rho_0)a^3g}{3k_BT}d\theta\right)\right]\)"
            ),
        ],
        "reference_answer": (
            r"\(z_i=(d+D)\theta+\frac{w\theta}{1+\xi C_0"
            r"\exp\!\left(-\frac{4\pi(\rho-\rho_0)a^3g}{3k_BT}d\theta\right)}\)"
        ),
        "curation_tier": "strict_clean",
        "reason": "Removed gold-only punctuation and repaired the malformed k_B subscript.",
    },
    "079657b3-e215-4944-8a67-8bb5347e4f15": {
        "display_gold": "<IUPAC>Nickel(III) oxide-hydroxide (NiO(OH))</IUPAC>",
        "display_distractors": [
            "<IUPAC>Nickel(III) hydroxide (Ni(OH)3)</IUPAC>",
            "<IUPAC>Nickel(III) oxide (Ni2O3)</IUPAC>",
            "<IUPAC>Cobalt(III) oxide-hydroxide (CoO(OH))</IUPAC>",
        ],
        "reference_answer": "<IUPAC>Nickel(III) oxide-hydroxide (NiO(OH))</IUPAC>",
        "curation_tier": "strict_clean",
        "reason": "Removed the other gold-only period and normalized the IUPAC wrapper.",
    },
}

PROBLEM_REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "bb0539ef-d9fd-4215-bf16-b0eca44a8778": (
        (
            r"\( 6.674 \times 10^{-11} N \bullet m^2/kg^2 \)",
            r"\( 6.674 \times 10^{-11} N \cdot m^2/kg^2 \)",
        ),
    ),
    "16318af5-cfa6-4b2c-bf25-9eb3ac468b48": (
        (
            "However, the distance between the two ends of the polymer, denoted as\n\n"
            "`\\( r≠L \\)`.",
            "The end-to-end distance `\\(r\\)` generally does not equal the contour "
            "length `\\(L\\)`: `\\(r \\ne L\\)`.",
        ),
    ),
    "328c6a47-15d7-470e-99f5-73e468634614": (
        (
            "where `\\( f \\)` is a function of `\\( \\phi_n,\\alpha \\)' and `\\( u \\)`.",
            "where `\\(f\\)` is a function of `\\(\\phi_n,\\alpha\\)`, and `\\(u\\)`.",
        ),
    ),
}

# Additional source concerns found during reconciliation with an independent
# review and then checked by a targeted high-reasoning audit. These affect only
# review-set membership: the supplied gold remains the scoring key in the
# complete 100-question release.
SUPPLEMENTAL_SOURCE_FLAGS: dict[str, dict[str, str]] = {
    "2c46387b-36e6-43cb-9e73-dbf1341bfafd": {
        "gold_verdict": "ambiguous",
        "gold_analysis": (
            "The value 1.56 follows if O is the fixed center of mass and P2 and P3 are "
            "mirror-symmetric about OP: OP=(2/3) PP2 cos(11.6 degrees), which gives "
            "OP2 about 92.8 L. Those standard figure-eight choreography assumptions are "
            "not stated explicitly, so the literal geometric data do not uniquely determine OP2."
        ),
        "stem_verdict": "ambiguous",
        "stem_analysis": (
            "An expert should determine whether the phrase 'eight-shaped orbit' conventionally "
            "imports the center-of-mass and reflection symmetries needed for the supplied answer."
        ),
    },
    "85b4f862-d881-4a79-8c5d-3e927b486b71": {
        "gold_verdict": "incorrect",
        "gold_analysis": (
            "The titration gives 0.003750 mol total Fe atoms. Solving x+y=0.225 g and "
            "x/55.845+2y/159.687=0.003750 gives about 0.17316 g Fe and 0.05184 g "
            "Fe2O3, or 76.96% Fe and 23.04% Fe2O3. The supplied 76.89%/23.11% does "
            "not follow from standard molar masses."
        ),
        "stem_verdict": "clear",
        "stem_analysis": "The data and requested rounding are sufficient; the issue is arithmetic.",
    },
    "fcbd76ad-1b45-4f40-ad9d-d2bc5016f02c": {
        "gold_verdict": "incorrect",
        "gold_analysis": (
            "The viability experiment under amino-acid deprivation is attributed to killed "
            "bacteria supplying additional nutrients, including amino acids, through lysosomal "
            "recycling. The source paper's antioxidant-metabolite/ROS result is a separate "
            "finding, so 'increased antioxidant availability' conflates the two conclusions."
        ),
        "stem_verdict": "clear",
        "stem_analysis": (
            "The amino-acid rescue described in the stem points to nutrient provision; no stem "
            "repair is needed if the reference answer is changed accordingly."
        ),
    },
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def clean_choice_markup(value: str) -> str:
    """Remove source-only Markdown code wrappers and surrounding whitespace."""
    value = value.strip().strip("`").strip()
    if re.match(r"\\{2,}[([]", value):
        value = re.sub(r"\\{2,}", r"\\", value)
    return value


def clean_smiles(value: str) -> str:
    """Remove Markdown-only escapes from bracket atoms."""
    return re.sub(r"\\+([\[\]])", r"\1", value)


def canonical_smiles(text: str) -> tuple[list[str], list[str]]:
    from rdkit import Chem

    canonical: list[str] = []
    failures: list[str] = []
    for value in extract_tagged_values(text, "SMILES"):
        # Dataset answers sometimes protect bracket atoms for Markdown as
        # ``\[N+\]``. Those backslashes are presentation escapes, unlike the
        # slash/backslash bond markers elsewhere in a SMILES string.
        parsed_value = clean_smiles(value)
        molecule = Chem.MolFromSmiles(parsed_value)
        if molecule is None:
            failures.append(parsed_value)
        else:
            canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return canonical, failures


def canonicalize_structure_choice(
    text: str,
    *,
    fallback_inchis: list[str] | None = None,
    fallback_smiles: list[str] | None = None,
) -> tuple[str, list[str], list[str]]:
    """Return uniform canonical SMILES display, with an InChI fallback for gold."""
    from rdkit import Chem

    values = extract_tagged_values(text, "SMILES")
    if not values:
        return text, [], []
    canonical: list[str] = []
    failures: list[str] = []
    fallbacks: list[str] = []
    for index, value in enumerate(values):
        parsed_value = clean_smiles(value)
        molecule = Chem.MolFromSmiles(parsed_value)
        if molecule is None and fallback_inchis and index < len(fallback_inchis):
            molecule = Chem.MolFromInchi(fallback_inchis[index])
            if molecule is not None:
                fallbacks.append(f"B{index + 1} from source InChI")
        if molecule is None and fallback_smiles and index < len(fallback_smiles):
            molecule = Chem.MolFromSmiles(fallback_smiles[index])
            if molecule is not None:
                fallbacks.append(f"B{index + 1} from unanimous distractor component")
        if molecule is None:
            failures.append(parsed_value)
            canonical.append(parsed_value)
        else:
            canonical.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    if len(canonical) == 1:
        return f"<SMILES>{canonical[0]}</SMILES>", failures, fallbacks
    return (
        "; ".join(
            f"B{index}: <SMILES>{value}</SMILES>" for index, value in enumerate(canonical, start=1)
        ),
        failures,
        fallbacks,
    )


def stable_seed(item_id: str) -> int:
    seed = int.from_bytes(
        hashlib.sha256(f"{item_id}:frontierscience-mc-v1".encode()).digest()[:8],
        byteorder="big",
    )
    return seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="triage_v2.jsonl")
    parser.add_argument("--original-triage", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-output", type=Path)
    parser.add_argument("--validation-output", type=Path, required=True)
    parser.add_argument("--flags-output", type=Path)
    parser.add_argument("--flags-report", type=Path)
    parser.add_argument("--require-rdkit", action="store_true")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    original_triage = (
        {record["item_id"]: record for record in load_jsonl(args.original_triage)}
        if args.original_triage
        else {}
    )
    # Preserve the existing answer-position assignment. Supplemental review
    # flags must not reshuffle choices in the released 100-question artifact.
    position_verified_ids = [
        record["item_id"]
        for record in records
        if record["tier"] != "blocked_source"
        or bool(MANUAL_OVERRIDES.get(record["item_id"], {}).get("usable"))
    ]
    position_flagged_ids = [
        record["item_id"] for record in records if record["item_id"] not in position_verified_ids
    ]
    position_verified_ids = sorted(
        position_verified_ids,
        key=lambda item_id: hashlib.sha256(f"{item_id}:correct-position-v1".encode()).digest(),
    )
    position_flagged_ids = sorted(
        position_flagged_ids,
        key=lambda item_id: hashlib.sha256(f"{item_id}:correct-position-v1".encode()).digest(),
    )
    # For 86 verified records this yields 22/22/21/21; the 14 flagged records
    # fill the complementary 3/3/4/4, so the complete set is exactly balanced.
    correct_positions = {item_id: rank % 4 for rank, item_id in enumerate(position_verified_ids)}
    correct_positions.update(
        {item_id: (rank + 2) % 4 for rank, item_id in enumerate(position_flagged_ids)}
    )
    output_records: list[dict[str, Any]] = []
    item_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    rdkit_available = True
    try:
        from rdkit import Chem  # noqa: F401
    except ImportError:
        rdkit_available = False
        if args.require_rdkit:
            raise

    for record in records:
        item_id = record["item_id"]
        override = MANUAL_OVERRIDES.get(item_id)
        problem = record["problem"]
        for source, replacement in PROBLEM_REPLACEMENTS.get(item_id, ()):
            if source not in problem:
                raise ValueError(f"problem replacement source not found for {item_id}")
            problem = problem.replace(source, replacement)
        original_source_flag = record["tier"] == "blocked_source" and not (
            override and override.get("usable")
        )
        trusted_source_flag = original_source_flag or item_id in SUPPLEMENTAL_SOURCE_FLAGS
        choice_source = original_triage[item_id] if original_source_flag else record
        raw_choices = [
            choice_source["display_gold"],
            *choice_source["display_distractors"],
        ]
        if override:
            raw_choices = [override["display_gold"], *override["display_distractors"]]
        raw_choices = [clean_choice_markup(choice) for choice in raw_choices]
        pre_errors: list[str] = []
        normalizations: list[str] = []
        has_smiles = any(extract_tagged_values(choice, "SMILES") for choice in raw_choices)
        if rdkit_available and has_smiles:
            fallback_inchis = extract_tagged_values(record["gold"], "INCHI")
            distractor_canonical = [canonical_smiles(choice) for choice in raw_choices[1:]]
            component_count = len(extract_tagged_values(raw_choices[0], "SMILES"))
            consensus: list[str] = []
            for component_index in range(component_count):
                values = [
                    canonical[component_index]
                    for canonical, failures in distractor_canonical
                    if not failures and component_index < len(canonical)
                ]
                consensus.append(values[0] if len(values) == 3 and len(set(values)) == 1 else "")
            normalized_choices: list[str] = []
            for index, choice in enumerate(raw_choices):
                normalized_choice, failures, fallbacks = canonicalize_structure_choice(
                    choice,
                    fallback_inchis=fallback_inchis if index == 0 else None,
                    fallback_smiles=consensus if index == 0 else None,
                )
                normalized_choices.append(normalized_choice)
                if failures:
                    pre_errors.append(f"invalid SMILES: {failures}")
                normalizations.extend(fallbacks)
            raw_choices = normalized_choices
        normalized = [normalized_text(choice) for choice in raw_choices]
        errors: list[str] = pre_errors
        warnings: list[str] = []
        if item_id in seen_ids:
            errors.append("duplicate item_id")
        seen_ids.add(item_id)
        if len(raw_choices) != 4:
            errors.append(f"expected 4 choices, found {len(raw_choices)}")
        if len(set(normalized)) != len(normalized):
            errors.append("textually duplicate choices")

        smiles_counts = [len(extract_tagged_values(choice, "SMILES")) for choice in raw_choices]
        if any(smiles_counts):
            if len(set(smiles_counts)) != 1:
                errors.append(f"inconsistent SMILES counts: {smiles_counts}")
            if rdkit_available:
                canonical_choices: list[tuple[str, ...]] = []
                for choice in raw_choices:
                    canonical, failures = canonical_smiles(choice)
                    if failures:
                        errors.append(f"invalid SMILES: {failures}")
                    canonical_choices.append(tuple(canonical))
                if len(set(canonical_choices)) != len(canonical_choices):
                    errors.append("chemically duplicate structural choices")

        lengths = [len(value) for value in normalized]
        shortest = min(lengths)
        longest = max(lengths)
        if shortest and longest / shortest > 4:
            warnings.append(f"choice-length ratio is {longest / shortest:.2f}")

        correct_index = correct_positions[item_id]
        distractors = raw_choices[1:]
        random.Random(stable_seed(item_id)).shuffle(distractors)
        shuffled = distractors[:]
        shuffled.insert(correct_index, raw_choices[0])
        usable = True if trusted_source_flag else record["tier"] != "blocked_source"
        if override and "usable" in override:
            usable = bool(override["usable"])
        if trusted_source_flag:
            curation_tier = "trusted_source_flagged"
        else:
            curation_tier = (
                override.get("curation_tier", record["tier"]) if override else record["tier"]
            )
        provenance = (
            override.get("provenance", choice_source["selected"])
            if override
            else choice_source["selected"]
        )
        source_answer_flag = None
        if trusted_source_flag:
            if item_id in SUPPLEMENTAL_SOURCE_FLAGS:
                source_answer_flag = dict(SUPPLEMENTAL_SOURCE_FLAGS[item_id])
            else:
                source_answer_flag = {
                    "gold_verdict": record["audit"]["gold_verdict"],
                    "gold_analysis": record["audit"]["gold_analysis"],
                    "stem_verdict": record["audit"]["stem_verdict"],
                    "stem_analysis": record["audit"]["stem_analysis"],
                }
            source_answer_flag["policy"] = "reference answer retained as authoritative"
        output_records.append(
            {
                "task_group_id": item_id,
                "subject": record["subject"],
                "problem": problem,
                "choices": shuffled,
                "correct_choice_index": correct_index,
                "correct_choice_label": CHOICE_LABELS[correct_index],
                # This is the canonical MC answer representation, including any
                # structure canonicalization or display repair applied above.
                "reference_answer": raw_choices[0],
                "curation_tier": curation_tier,
                "prior_tier": record.get("prior_tier"),
                "usable": usable,
                "distractor_provenance": provenance,
                "audit": record.get("audit"),
                "display_override": override,
                "source_answer_flag": source_answer_flag,
            }
        )
        item_reports.append(
            {
                "item_id": item_id,
                "subject": record["subject"],
                "tier": curation_tier,
                "usable": usable,
                "errors": errors,
                "warnings": warnings,
                "normalizations": normalizations,
                "smiles_per_choice": smiles_counts,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    verified_records = [
        record for record in output_records if record["curation_tier"] != "trusted_source_flagged"
    ]
    if args.verified_output:
        with args.verified_output.open("w") as handle:
            for record in verified_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    error_count = sum(len(record["errors"]) for record in item_reports)
    validation = {
        "item_count": len(records),
        "unique_item_count": len(seen_ids),
        "usable_count": sum(record["usable"] for record in item_reports),
        "blocked_count": sum(not record["usable"] for record in item_reports),
        "verified_count": len(verified_records),
        "full_count": len(output_records),
        "rdkit_available": rdkit_available,
        "error_count": error_count,
        "warning_count": sum(len(record["warnings"]) for record in item_reports),
        "normalization_count": sum(len(record["normalizations"]) for record in item_reports),
        "items_with_findings": [
            record
            for record in item_reports
            if record["errors"] or record["warnings"] or record["normalizations"]
        ],
    }
    args.validation_output.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    flagged = [record for record in output_records if record["source_answer_flag"]]
    if args.flags_output:
        with args.flags_output.open("w") as handle:
            for record in flagged:
                flag_record = {
                    "task_group_id": record["task_group_id"],
                    "subject": record["subject"],
                    "reference_answer": record["reference_answer"],
                    **record["source_answer_flag"],
                }
                handle.write(json.dumps(flag_record, ensure_ascii=False) + "\n")
    if args.flags_report:
        lines = [
            "# FrontierScience source-answer flags",
            "",
            (
                "These flags do not alter evaluation keys. The supplied reference answers remain "
                "authoritative."
            ),
            "",
        ]
        for record in flagged:
            flag = record["source_answer_flag"]
            lines.extend(
                [
                    f"## `{record['task_group_id']}` ({record['subject']})",
                    "",
                    f"Gold assessment: **{flag['gold_verdict']}**",
                    "",
                    flag["gold_analysis"],
                    "",
                ]
            )
        args.flags_report.write_text("\n".join(lines))
    if error_count:
        raise SystemExit(f"validation failed with {error_count} error(s)")


if __name__ == "__main__":
    main()
