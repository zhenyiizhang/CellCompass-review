from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"


GENE_META = {
    "DKK1": {
        "axis": "demethylation/reactivation",
        "source_role": "source-paper tumor-suppressor reactivation marker",
        "expected": "up",
        "claim": "supports 5-AZA-CdR demethylation/reactivation arm",
        "boundary": "consistent with source paper, not a new mechanism by itself",
    },
    "GAS5": {
        "axis": "demethylation/reactivation",
        "source_role": "source-paper tumor-suppressor-associated lncRNA reactivation marker",
        "expected": "up",
        "claim": "supports early growth-arrest/tumor-suppressor-like reactivation",
        "boundary": "use as source-aligned evidence, not causal proof",
    },
    "EREG": {
        "axis": "oncogene/proliferation repression",
        "source_role": "source-paper reported cancer-associated gene down after treatment",
        "expected": "down",
        "claim": "partial support for growth-factor/proliferation-associated repression",
        "boundary": "total/old decrease is clearer than new RNA decrease",
    },
    "MALAT1": {
        "axis": "caution",
        "source_role": "source-paper reported cancer-associated lncRNA down after treatment",
        "expected": "down",
        "claim": "not reproduced locally; keep out of positive claim",
        "boundary": "do not claim MALAT1 repression from current local table",
    },
    "STAT1": {
        "axis": "viral-mimicry/interferon boundary",
        "source_role": "source-paper STAT1 activation/regulon axis",
        "expected": "up",
        "claim": "weak local gene-expression support for interferon axis",
        "boundary": "regulon/protein activation needs dedicated analysis",
    },
    "HEYL": {
        "axis": "regulon boundary",
        "source_role": "source-paper regulon restoration axis",
        "expected": "up",
        "claim": "source-paper anchor only in current analysis",
        "boundary": "do not center without recomputed regulon activity",
    },
    "PITX1": {
        "axis": "regulon boundary",
        "source_role": "source-paper regulon restoration axis",
        "expected": "up",
        "claim": "source-paper anchor only in current analysis",
        "boundary": "do not center without recomputed regulon activity",
    },
    "B2M": {
        "axis": "viral-mimicry/interferon-like surface",
        "source_role": "PulseBridge immune-surface readout connected to source-paper viral-mimicry logic",
        "expected": "up",
        "claim": "supports antigen-presentation/surface remodeling along response route",
        "boundary": "consistent with interferon biology, not direct STAT1 proof",
    },
    "HLA-A": {
        "axis": "viral-mimicry/interferon-like surface",
        "source_role": "PulseBridge antigen-presentation readout",
        "expected": "up",
        "claim": "supports immune-surface remodeling",
        "boundary": "not a direct immune killing readout",
    },
    "HLA-B": {
        "axis": "viral-mimicry/interferon-like surface",
        "source_role": "PulseBridge antigen-presentation readout",
        "expected": "up",
        "claim": "supports immune-surface remodeling",
        "boundary": "not a direct immune killing readout",
    },
    "HLA-C": {
        "axis": "viral-mimicry/interferon-like surface",
        "source_role": "PulseBridge antigen-presentation readout",
        "expected": "up",
        "claim": "supports immune-surface remodeling",
        "boundary": "not a direct immune killing readout",
    },
    "IFITM3": {
        "axis": "viral-mimicry/interferon-like surface",
        "source_role": "PulseBridge antiviral/interferon-like readout",
        "expected": "up",
        "claim": "supports antiviral membrane remodeling consistent with viral mimicry",
        "boundary": "interferon causality needs perturbation/regulon evidence",
    },
    "SAT1": {
        "axis": "polyamine/metabolic response",
        "source_role": "PulseBridge-nominated metabolic response gene",
        "expected": "up",
        "claim": "new hypothesis: polyamine catabolism/remodeling is part of late response",
        "boundary": "hypothesis-generating, not causal driver evidence",
    },
    "LGALS1": {
        "axis": "adaptive adhesion/escape candidate",
        "source_role": "PulseBridge-nominated adhesion/glycan stress gene",
        "expected": "up",
        "claim": "new hypothesis: possible adaptive adhesion/immune-evasion-like branch",
        "boundary": "do not frame as beneficial anti-tumor marker",
    },
    "PCNA": {
        "axis": "proliferation repression",
        "source_role": "PulseBridge proliferation readout",
        "expected": "down",
        "claim": "supports reduced proliferation/DNA-replication-associated expression",
        "boundary": "does not prove cell death or growth-rate reduction",
    },
    "HSP90AB1": {
        "axis": "proteostasis remodeling",
        "source_role": "PulseBridge heat-shock/proteostasis readout",
        "expected": "down",
        "claim": "supports reduced proteostasis demand along treated state",
        "boundary": "supportive readout, not a source-paper main marker",
    },
    "HSP90AA1": {
        "axis": "proteostasis remodeling",
        "source_role": "PulseBridge heat-shock/proteostasis readout",
        "expected": "down",
        "claim": "supports reduced proteostasis demand along treated state",
        "boundary": "supportive readout, not a source-paper main marker",
    },
    "HSPD1": {
        "axis": "proteostasis remodeling",
        "source_role": "PulseBridge mitochondrial/proteostasis readout",
        "expected": "down",
        "claim": "supports proteostasis remodeling",
        "boundary": "supportive readout, not a source-paper main marker",
    },
    "FTL": {
        "axis": "iron/redox remodeling",
        "source_role": "PulseBridge iron/redox readout",
        "expected": "down",
        "claim": "supports iron/redox remodeling in treated state",
        "boundary": "supportive readout, not causal evidence",
    },
    "MT-CO1": {
        "axis": "mitochondrial remodeling",
        "source_role": "PulseBridge mitochondrial remodeling and PCIE-aligned gene",
        "expected": "up",
        "claim": "supports mitochondrial/metabolic remodeling and pulse-direction alignment",
        "boundary": "adaptive/stress readout, not necessarily beneficial",
    },
    "MT-CO2": {
        "axis": "mitochondrial remodeling",
        "source_role": "PulseBridge mitochondrial remodeling and PCIE-aligned gene",
        "expected": "up",
        "claim": "supports mitochondrial/metabolic remodeling and pulse-direction alignment",
        "boundary": "adaptive/stress readout, not necessarily beneficial",
    },
    "MT-CO3": {
        "axis": "mitochondrial remodeling",
        "source_role": "PulseBridge mitochondrial remodeling and PCIE-aligned gene",
        "expected": "up",
        "claim": "supports mitochondrial/metabolic remodeling and pulse-direction alignment",
        "boundary": "adaptive/stress readout, not necessarily beneficial",
    },
    "MT-ND4": {
        "axis": "mitochondrial remodeling",
        "source_role": "PulseBridge mitochondrial remodeling readout",
        "expected": "up",
        "claim": "supports mitochondrial/metabolic remodeling",
        "boundary": "adaptive/stress readout, not necessarily beneficial",
    },
    "MT-CYB": {
        "axis": "mitochondrial remodeling",
        "source_role": "PulseBridge mitochondrial remodeling readout",
        "expected": "up",
        "claim": "supports mitochondrial/metabolic remodeling",
        "boundary": "adaptive/stress readout, not necessarily beneficial",
    },
}


MODULE_TIMING = {
    "demethylation/reactivation": "tumor_suppressor_reactivation",
    "oncogene/proliferation repression": "oncogene_proliferation_repression",
    "proliferation repression": "oncogene_proliferation_repression",
    "viral-mimicry/interferon-like surface": "antigen_presentation_surface",
    "polyamine/metabolic response": "sat1_polyamine_catabolism",
    "adaptive adhesion/escape candidate": "lgals1_adhesion_escape",
    "mitochondrial remodeling": "mitochondrial_remodeling",
    "proteostasis remodeling": "proteostasis_iron_reduction",
    "iron/redox remodeling": "proteostasis_iron_reduction",
}


AXIS_SUMMARY = [
    {
        "axis": "demethylation/reactivation",
        "genes": "DKK1;GAS5",
        "source_connection": "directly source-paper-aligned",
        "biology_conclusion": "5-AZA-CdR response includes tumor-suppressor-associated reactivation, especially visible in newly synthesized RNA.",
        "claim_strength": "strong support",
    },
    {
        "axis": "oncogene/proliferation repression",
        "genes": "EREG;PCNA",
        "source_connection": "EREG source-aligned; PCNA PulseBridge proliferation readout",
        "biology_conclusion": "treated route contains a growth-factor/proliferation-associated repression component.",
        "claim_strength": "moderate support",
    },
    {
        "axis": "viral-mimicry/interferon-like surface",
        "genes": "B2M;HLA-A;HLA-B;HLA-C;IFITM3;STAT1",
        "source_connection": "consistent with source-paper STAT1/viral-mimicry interpretation",
        "biology_conclusion": "late response states show immune-surface and antiviral-like remodeling.",
        "claim_strength": "consistent but indirect",
    },
    {
        "axis": "polyamine/metabolic response",
        "genes": "SAT1",
        "source_connection": "new PulseBridge-nominated biology",
        "biology_conclusion": "trajectory nominates polyamine catabolism/remodeling as a late drug-response component.",
        "claim_strength": "new hypothesis",
    },
    {
        "axis": "mitochondrial remodeling",
        "genes": "MT-CO1;MT-CO2;MT-CO3;MT-ND4;MT-CYB;MT-RNR2",
        "source_connection": "new PulseBridge-nominated and PCIE-aligned biology",
        "biology_conclusion": "metabolic/mitochondrial axes carry part of the pulse-aligned response direction.",
        "claim_strength": "new hypothesis with direction support",
    },
    {
        "axis": "adaptive adhesion/escape candidate",
        "genes": "LGALS1",
        "source_connection": "new PulseBridge caution axis",
        "biology_conclusion": "trajectory flags a possible adhesion/immune-evasion-like adaptive branch.",
        "claim_strength": "new cautionary hypothesis",
    },
    {
        "axis": "MALAT1 caution",
        "genes": "MALAT1",
        "source_connection": "source-paper marker not reproduced locally",
        "biology_conclusion": "current local PulseBridge tables do not support using MALAT1 repression as evidence.",
        "claim_strength": "negative/caution",
    },
    {
        "axis": "STAT1/HEYL/PITX1 regulon boundary",
        "genes": "STAT1;HEYL;PITX1",
        "source_connection": "source-paper regulon/protein claim",
        "biology_conclusion": "current gene-expression readout is weaker than the source-paper regulon/protein evidence.",
        "claim_strength": "needs regulon recomputation",
    },
]


def signed_support(delta: float, expected: str) -> float:
    if not np.isfinite(delta):
        return np.nan
    if expected == "down":
        return -delta
    return delta


def day_delta(gene_day: pd.DataFrame, gene: str, layer: str) -> tuple[float, float, float]:
    sub = gene_day[(gene_day["gene"] == gene) & (gene_day["layer"] == layer)].sort_values("time")
    if sub.empty:
        return np.nan, np.nan, np.nan
    day0 = sub.loc[sub["time"] == 0.0, "mean_value"]
    day3 = sub.loc[sub["time"] == 3.0, "mean_value"]
    if day0.empty or day3.empty:
        return np.nan, np.nan, np.nan
    delta = float(day3.iloc[0] - day0.iloc[0])
    peak_row = sub.iloc[(sub["mean_value"] - float(day0.iloc[0])).abs().argmax()]
    return delta, float(peak_row["time"]), float(peak_row["mean_value"] - day0.iloc[0])


def build_gene_matrix() -> pd.DataFrame:
    gene_day = pd.read_csv(TAB / "targeted_drug_response_gene_by_day_layer.csv")
    original = pd.read_csv(TAB / "new_original_paper_marker_readout_by_layer.csv")
    pcie = pd.read_csv(TAB / "pcie_gene_space_top_genes.csv")
    pcie_random = pd.read_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv")
    timing = pd.read_csv(TAB / "pulsebridge_rollout_biology_event_timing.csv")

    rows = []
    for gene, meta in GENE_META.items():
        source = gene_day if gene in set(gene_day["gene"]) else original
        total_delta, peak_time, peak_delta = day_delta(source, gene, "log1p_norm")
        new_delta, _, _ = day_delta(source, gene, "new")
        old_delta, _, _ = day_delta(source, gene, "old")
        pcie_row = pcie[pcie["gene"] == gene]
        random_row = pcie_random[pcie_random["gene"] == gene]
        module = MODULE_TIMING.get(meta["axis"])
        module_row = timing[timing["module"] == module] if module else pd.DataFrame()

        support_total = signed_support(total_delta, meta["expected"])
        support_new = signed_support(new_delta, meta["expected"])
        if meta["expected"] == "down":
            local_status = "supports" if support_total > 0 else "not_reproduced"
        else:
            local_status = "supports" if support_total > 0 else "not_reproduced"
        if gene in {"STAT1", "HEYL", "PITX1"}:
            local_status = "weak_or_boundary"
        if gene == "MALAT1":
            local_status = "not_reproduced"

        rows.append(
            {
                "gene": gene,
                "biology_axis": meta["axis"],
                "source_paper_role_or_connection": meta["source_role"],
                "expected_direction": meta["expected"],
                "total_day3_minus_day0": total_delta,
                "new_day3_minus_day0": new_delta,
                "old_day3_minus_day0": old_delta,
                "expected_signed_total_support": support_total,
                "expected_signed_new_support": support_new,
                "observed_peak_day": peak_time,
                "observed_peak_delta_from_day0": peak_delta,
                "pulsebridge_module": module if module else "",
                "pulsebridge_peak_time": float(module_row["peak_time"].iloc[0]) if len(module_row) else np.nan,
                "pulsebridge_peak_signed_delta": float(module_row["peak_signed_delta"].iloc[0]) if len(module_row) else np.nan,
                "pcie_same_sign_fraction": float(pcie_row["same_sign_fraction"].iloc[0]) if len(pcie_row) else np.nan,
                "pcie_direction_class": str(pcie_row["direction_class"].iloc[0]) if len(pcie_row) else "",
                "random_same_sign_fraction": float(random_row["random_same"].iloc[0]) if len(random_row) else np.nan,
                "local_support_status": local_status,
                "biology_claim": meta["claim"],
                "claim_boundary": meta["boundary"],
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "final_gene_level_biology_response_matrix.csv", index=False)

    axis_rows = []
    for item in AXIS_SUMMARY:
        genes = item["genes"].split(";")
        sub = out[out["gene"].isin(genes)]
        axis_rows.append(
            {
                **item,
                "mean_expected_signed_total_support": float(sub["expected_signed_total_support"].mean(skipna=True)),
                "mean_expected_signed_new_support": float(sub["expected_signed_new_support"].mean(skipna=True)),
                "representative_pulsebridge_peak_time": float(sub["pulsebridge_peak_time"].dropna().iloc[0])
                if sub["pulsebridge_peak_time"].notna().any()
                else np.nan,
                "genes_supported_locally": ";".join(sub.loc[sub["local_support_status"] == "supports", "gene"].tolist()),
                "genes_boundary_or_not_reproduced": ";".join(
                    sub.loc[sub["local_support_status"].isin(["not_reproduced", "weak_or_boundary"]), "gene"].tolist()
                ),
            }
        )
    axis_out = pd.DataFrame(axis_rows)
    axis_out.to_csv(TAB / "final_biology_axis_response_matrix.csv", index=False)
    return out


def plot_gene_evidence(out: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
        }
    )

    fig = plt.figure(figsize=(12.5, 8.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], wspace=0.34, hspace=0.42)

    ax1 = fig.add_subplot(gs[0, 0])
    source_genes = ["DKK1", "GAS5", "EREG", "MALAT1", "STAT1", "HEYL", "PITX1"]
    sub = out[out["gene"].isin(source_genes)].set_index("gene").loc[source_genes].reset_index()
    y = np.arange(len(sub))
    for layer, offset, color in [
        ("total_day3_minus_day0", -0.22, "#365C8D"),
        ("new_day3_minus_day0", 0.0, "#2A9D8F"),
        ("old_day3_minus_day0", 0.22, "#B56576"),
    ]:
        vals = sub[layer].to_numpy(float)
        ax1.barh(y + offset, vals, height=0.18, color=color, label=layer.replace("_day3_minus_day0", ""))
    ax1.axvline(0, color="#222222", lw=0.8)
    ax1.set_yticks(y)
    ax1.set_yticklabels(sub["gene"])
    ax1.invert_yaxis()
    ax1.set_xlabel("day3 - day0 expression")
    ax1.set_title("a. Source-paper genes: layer-specific local readout")
    ax1.set_xlim(-0.7, 3.65)
    ax1.legend(frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=3, fontsize=7)
    for i, row in sub.iterrows():
        status = row["local_support_status"]
        label = "support" if status == "supports" else ("caution" if status == "not_reproduced" else "weak")
        color = "#2A9D8F" if label == "support" else ("#C43B3B" if label == "caution" else "#9A7D0A")
        ax1.text(
            3.48,
            i,
            label,
            ha="right",
            va="center",
            color=color,
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.6},
        )

    ax2 = fig.add_subplot(gs[0, 1])
    axis_order = [
        "demethylation/reactivation",
        "oncogene/proliferation repression",
        "viral-mimicry/interferon-like surface",
        "polyamine/metabolic response",
        "mitochondrial remodeling",
        "adaptive adhesion/escape candidate",
        "proteostasis remodeling",
        "iron/redox remodeling",
    ]
    axis_df = (
        out[out["biology_axis"].isin(axis_order)]
        .groupby("biology_axis", as_index=False)
        .agg(total_support=("expected_signed_total_support", "mean"), new_support=("expected_signed_new_support", "mean"))
    )
    axis_df = axis_df.set_index("biology_axis").reindex(axis_order).dropna(how="all").reset_index()
    y2 = np.arange(len(axis_df))
    ax2.barh(y2 - 0.16, axis_df["total_support"], height=0.28, color="#365C8D", label="total")
    ax2.barh(y2 + 0.16, axis_df["new_support"], height=0.28, color="#2A9D8F", label="new")
    ax2.axvline(0, color="#222222", lw=0.8)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(
        [
            "DKK1/GAS5\nreactivation",
            "EREG/PCNA\nrepression",
            "B2M/HLA/IFITM3\nimmune-like",
            "SAT1\npolyamine",
            "MT genes\nmitochondrial",
            "LGALS1\nadaptive",
            "HSP\nproteostasis",
            "FTL\niron/redox",
        ][: len(axis_df)]
    )
    ax2.invert_yaxis()
    ax2.set_xlabel("expected-signed day3 - day0 support")
    ax2.set_title("b. Gene axes: positive means supports the proposed direction")
    ax2.legend(frameon=False, loc="lower right", fontsize=7)

    ax3 = fig.add_subplot(gs[1, 0])
    timing = out.dropna(subset=["pulsebridge_peak_time"]).drop_duplicates("pulsebridge_module")
    timing = timing.sort_values("pulsebridge_peak_time")
    labels = []
    for _, row in timing.iterrows():
        module = row["pulsebridge_module"]
        labels.append(
            {
                "tumor_suppressor_reactivation": "DKK1/GAS5",
                "oncogene_proliferation_repression": "EREG/PCNA",
                "antigen_presentation_surface": "B2M/HLA",
                "sat1_polyamine_catabolism": "SAT1",
                "lgals1_adhesion_escape": "LGALS1",
                "mitochondrial_remodeling": "MT genes",
                "proteostasis_iron_reduction": "HSP/FTL",
            }.get(module, module)
        )
    y3 = np.arange(len(timing))
    ax3.hlines(y3, 0, timing["pulsebridge_peak_time"], color="#A7A7A7", lw=1.2)
    ax3.scatter(timing["pulsebridge_peak_time"], y3, s=70, c=timing["pulsebridge_peak_signed_delta"], cmap="viridis")
    ax3.set_yticks(y3)
    ax3.set_yticklabels(labels)
    ax3.invert_yaxis()
    ax3.set_xlim(0, 3.05)
    ax3.set_xlabel("PulseBridge inferred peak day")
    ax3.set_title("c. Trajectory timing: when each biology axis appears")

    ax4 = fig.add_subplot(gs[1, 1])
    pcie_genes = ["MT-RNR2", "RPS2", "MT-CO1", "MALAT1", "SAT1", "LGALS1"]
    pcie_small = pd.read_csv(TAB / "final_pcie_real_vs_random_gene_alignment.csv")
    pcie_small = pcie_small.set_index("gene").reindex(pcie_genes).dropna().reset_index()
    y4 = np.arange(len(pcie_small))
    ax4.hlines(y4, pcie_small["random_same"] * 100, pcie_small["real_same"] * 100, color="#BBBBBB", lw=1.5)
    ax4.scatter(pcie_small["random_same"] * 100, y4, color="#777777", s=38, label="random")
    ax4.scatter(pcie_small["real_same"] * 100, y4, color="#D55E00", s=48, label="PulseBridge")
    ax4.set_yticks(y4)
    ax4.set_yticklabels(pcie_small["gene"])
    ax4.invert_yaxis()
    ax4.set_xlim(30, 105)
    ax4.set_xlabel("same-sign agreement with metabolic pulse direction (%)")
    ax4.set_title("d. Gene-space direction: real velocity versus random")
    ax4.legend(frameon=False, loc="lower right", fontsize=7)

    fig.suptitle("Gene-level evidence matrix for the PulseBridge HCT116 5-AZA-CdR story", fontsize=13, y=0.99)
    fig.savefig(FIG / "story_gene_level_biology_evidence_matrix.png", bbox_inches="tight")
    fig.savefig(FIG / "story_gene_level_biology_evidence_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out = build_gene_matrix()
    plot_gene_evidence(out)
    print(TAB / "final_gene_level_biology_response_matrix.csv")
    print(TAB / "final_biology_axis_response_matrix.csv")
    print(FIG / "story_gene_level_biology_evidence_matrix.png")


if __name__ == "__main__":
    main()
