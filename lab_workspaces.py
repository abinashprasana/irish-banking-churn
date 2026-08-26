"""Premium, evidence-led Streamlit workspaces for Atlantic Ledger.

The functions in this module are intentionally presentation-only. Model training,
prediction, policy rules, and agent contracts remain owned by their existing
modules.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve


PlotStyler = Callable[..., None]


def _fact_rail(items: list[dict[str, str]]) -> None:
    cells = []
    for item in items:
        tone = escape(item.get("tone", ""))
        cells.append(
            '<div class="fact-rail__item ' + tone + '">'
            f'<span class="fact-rail__label">{escape(item["label"])}</span>'
            f'<strong class="fact-rail__value">{escape(item["value"])}</strong>'
            f'<span class="fact-rail__note">{escape(item["note"])}</span>'
            "</div>"
        )
    st.markdown(
        '<section class="fact-rail" aria-label="Key evidence">'
        + "".join(cells)
        + "</section>",
        unsafe_allow_html=True,
    )


def _workspace_intro(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        '<header class="workspace-intro">'
        f'<span class="workspace-intro__eyebrow">{escape(eyebrow)}</span>'
        f'<h2>{escape(title)}</h2>'
        f'<p>{escape(description)}</p>'
        "</header>",
        unsafe_allow_html=True,
    )


def render_data_limits(
    *,
    data: pd.DataFrame,
    feature_count: int,
    style_plot: PlotStyler,
    churn_color: str,
    retain_color: str,
) -> None:
    """Render provenance, market context, and one focused segment view."""

    _workspace_intro(
        "Provenance workspace",
        "Data, context and limits",
        "Inspect what the demonstration contains, where its context comes from, and where its evidence must stop.",
    )

    _fact_rail(
        [
            {
                "label": "Dataset",
                "value": f"{len(data):,}",
                "note": "synthetic customer records",
            },
            {
                "label": "Model contract",
                "value": str(feature_count),
                "note": "ordered inputs checked at runtime",
            },
            {
                "label": "Evaluation",
                "value": "80 / 20",
                "note": "stratified train and holdout split",
                "tone": "approval",
            },
            {
                "label": "Intended use",
                "value": "Research",
                "note": "not a customer decision system",
                "tone": "review",
            },
        ]
    )

    st.markdown("### Irish banking context")
    _fact_rail(
        [
            {
                "label": "Accounts closed",
                "value": "1.17M",
                "note": "recorded by the Central Bank by June 2023",
            },
            {
                "label": "Switching difficulty",
                "value": "60%",
                "note": "reported in the cited CCPC research",
                "tone": "review",
            },
            {
                "label": "Local branch mattered",
                "value": "28%",
                "note": "among surveyed customers who switched",
                "tone": "approval",
            },
        ]
    )
    st.markdown(
        '<div class="evidence-note"><strong>Context, not training truth.</strong>'
        " The market exit explains why switching and service signals are studied. "
        "The customer rows and churn outcomes remain generated research data.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Published sources: [Central Bank of Ireland account migration statistics]"
        "(https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/account-migration-statistics) "
        "and [CCPC switching research]"
        "(https://www.ccpc.ie/about-us/advocacy-and-research/research/publication-details/"
        "ccpc-switching-research-%28phase-2%29)."
    )

    st.markdown("### Explore one segment relationship")
    st.caption(
        "Choose one view at a time. Every chart describes the synthetic dataset and does not establish causation."
    )
    segment_view = st.segmented_control(
        "Segment view",
        ["Account type", "Products held", "Tenure", "Complaints"],
        default="Account type",
        key="data_limits_segment_view",
        label_visibility="collapsed",
        width="stretch",
    )

    accessible_table: pd.DataFrame
    interpretation: str
    if segment_view == "Products held":
        accessible_table = data.groupby("num_products")["churn"].mean().mul(100).reset_index()
        accessible_table.columns = ["Products held", "Churn rate (%)"]
        fig = px.bar(
            accessible_table,
            x="Products held",
            y="Churn rate (%)",
            title="Observed churn by number of products",
            color_discrete_sequence=[churn_color],
        )
        interpretation = (
            "Observed churn falls as product count rises in the generated records. "
            "This is an association in the study, not evidence that adding a product prevents churn."
        )
    elif segment_view == "Tenure":
        chart_data = data[["tenure_months", "churn"]].copy()
        chart_data["Status"] = chart_data["churn"].map({0: "Retained", 1: "Churned"})
        fig = px.histogram(
            chart_data,
            x="tenure_months",
            color="Status",
            barmode="overlay",
            labels={"tenure_months": "Tenure (months)", "count": "Customers"},
            title="Tenure distribution by observed outcome",
            color_discrete_map={"Retained": retain_color, "Churned": churn_color},
        )
        fig.update_traces(opacity=0.78)
        accessible_table = (
            chart_data.assign(Tenure_band=pd.cut(chart_data["tenure_months"], bins=[0, 12, 36, 60, 120, 180]))
            .groupby("Tenure_band", observed=True)["churn"]
            .agg(["count", "mean"])
            .reset_index()
        )
        accessible_table["mean"] = accessible_table["mean"].mul(100)
        accessible_table.columns = ["Tenure band", "Records", "Churn rate (%)"]
        interpretation = (
            "Churned records are concentrated at shorter tenures. The pattern is descriptive and is not a production threshold."
        )
    elif segment_view == "Complaints":
        accessible_table = data.groupby("has_complaint_history")["churn"].mean().mul(100).reset_index()
        accessible_table["has_complaint_history"] = accessible_table["has_complaint_history"].map(
            {True: "Complaint on record", False: "No complaint"}
        )
        accessible_table.columns = ["Complaint status", "Churn rate (%)"]
        fig = px.bar(
            accessible_table,
            x="Complaint status",
            y="Churn rate (%)",
            color="Complaint status",
            title="Observed churn by complaint history",
            color_discrete_map={"Complaint on record": churn_color, "No complaint": retain_color},
        )
        fig.update_layout(showlegend=False)
        interpretation = (
            "Generated records with a complaint have a higher observed churn rate, making service recovery useful to inspect."
        )
    else:
        accessible_table = data.groupby("account_type")["churn"].mean().mul(100).reset_index()
        accessible_table.columns = ["Account type", "Churn rate (%)"]
        fig = px.bar(
            accessible_table,
            x="Account type",
            y="Churn rate (%)",
            title="Observed churn by account type",
            color_discrete_sequence=[churn_color],
        )
        interpretation = (
            "Current- or savings-only records show higher observed churn than mortgage records in this generated dataset."
        )

    style_plot(fig, height=410)
    fig.update_layout(showlegend=segment_view == "Tenure")
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
        key=f"segment_chart_{str(segment_view).lower().replace(' ', '_')}",
    )
    st.markdown(
        f'<p class="chart-interpretation"><strong>Reading:</strong> {escape(interpretation)}</p>',
        unsafe_allow_html=True,
    )
    with st.expander("View chart data"):
        st.dataframe(accessible_table.round(2), hide_index=True, width="stretch")

    with st.expander("Method and phase handoff"):
        st.markdown(
            f"""
            **Predict.** XGBoost receives the same {feature_count} ordered fields stored with the model. SMOTEENN is applied only to training data.

            **Explain.** SHAP identifies model contributions. DiCE can generate bounded candidate input scenarios for eligible scores.

            **Govern.** Product and segment evidence pass through a deterministic rule gate before recommendation formatting. A person remains responsible for any real-world action.
            """
        )

    with st.expander("Governance references"):
        st.markdown(
            "[EU AI Act, Article 86](https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_86) "
            "describes a right to a meaningful explanation in specified circumstances. This prototype does not "
            "claim to fall within Article 86, and its SHAP or DiCE views do not demonstrate compliance.\n\n"
            "[EBA guidelines on internal governance](https://www.eba.europa.eu/activities/single-rulebook/"
            "regulatory-activities/internal-governance/guidelines-internal-governance-under-crd) address "
            "responsibilities, risk management, and internal controls within their stated scope; they do not "
            "prescribe this retention workflow."
        )

    st.markdown(
        '<section class="hard-boundary" role="note"><span>Use boundary</span>'
        "<strong>Not for real customer decisions.</strong>"
        "<p>No record, score, explanation, candidate scenario, or governed recommendation shown here belongs to a real customer.</p>"
        "</section>",
        unsafe_allow_html=True,
    )


def render_model_evidence(
    *,
    y_test: pd.Series,
    y_prob,
    confusion,
    f1: float,
    roc_auc: float,
    average_precision: float,
    holdout_size: int,
    style_plot: PlotStyler,
    assets_dir: Path,
) -> None:
    """Render a focused validation workspace with one primary chart."""

    _workspace_intro(
        "Validation workspace",
        "Model evidence",
        "Review holdout performance, inspect the strongest model signals, and keep the evidence boundary visible.",
    )
    st.markdown(
        '<div class="evidence-note"><strong>Held-out evaluation.</strong>'
        " Every score below comes from the stratified 20% sample that was not used to fit XGBoost. "
        "Logistic Regression and Random Forest were retained as training benchmarks.</div>",
        unsafe_allow_html=True,
    )

    _fact_rail(
        [
            {
                "label": "Primary metric",
                "value": f"{average_precision:.3f}",
                "note": "average precision on the holdout sample",
                "tone": "approval",
            },
            {
                "label": "F1 score",
                "value": f"{f1:.3f}",
                "note": "balance of precision and recall",
            },
            {
                "label": "ROC AUC",
                "value": f"{roc_auc:.3f}",
                "note": "class separation across thresholds",
            },
            {
                "label": "Holdout",
                "value": f"{holdout_size:,}",
                "note": "synthetic records kept out of training",
            },
        ]
    )

    st.markdown("### Inspect one performance view")
    chart_view = st.segmented_control(
        "Performance view",
        ["Precision–recall", "Confusion matrix", "ROC"],
        default="Precision–recall",
        key="model_evidence_chart_view",
        label_visibility="collapsed",
        width="stretch",
    )

    if chart_view == "Confusion matrix":
        fig = px.imshow(
            confusion,
            text_auto=True,
            x=["Predicted: retained", "Predicted: churned"],
            y=["Actual: retained", "Actual: churned"],
            labels={"x": "Predicted", "y": "Actual", "color": "Records"},
            color_continuous_scale=[[0, "#F4F1E8"], [1, "#245B78"]],
            title="Classification outcomes at the configured threshold",
        )
        interpretation = (
            f"{int(confusion[1, 1])} of {int(confusion[1].sum())} actual churn records were identified; "
            f"{int(confusion[1, 0])} were missed."
        )
        accessible = pd.DataFrame(
            confusion,
            index=["Actual retained", "Actual churned"],
            columns=["Predicted retained", "Predicted churned"],
        )
    elif chart_view == "ROC":
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig = px.line(
            x=fpr,
            y=tpr,
            labels={"x": "False positive rate", "y": "True positive rate"},
            title=f"ROC curve · AUC {roc_auc:.3f}",
        )
        fig.update_traces(line_color="#245B78", line_width=3)
        fig.add_shape(type="line", line=dict(dash="dash", color="#82949C"), x0=0, x1=1, y0=0, y1=1)
        interpretation = "The curve remains well above the random-chance diagonal across most operating points."
        accessible = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
    else:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        prevalence = float(y_test.mean())
        fig = px.line(
            x=recall,
            y=precision,
            labels={"x": "Recall", "y": "Precision"},
            title=f"Precision–recall curve · AP {average_precision:.3f}",
        )
        fig.update_traces(line_color="#147D64", line_width=3)
        fig.add_shape(
            type="line",
            line=dict(dash="dash", color="#A66F20"),
            x0=0,
            x1=1,
            y0=prevalence,
            y1=prevalence,
        )
        interpretation = (
            f"The model remains above the {prevalence:.0%} class-prevalence baseline across most recall levels."
        )
        accessible = pd.DataFrame({"Recall": recall, "Precision": precision})

    style_plot(fig, height=430)
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False, "displaylogo": False, "responsive": True},
        key=f"evidence_chart_{str(chart_view).lower().replace('–', '_').replace(' ', '_')}",
    )
    st.markdown(
        f'<p class="chart-interpretation"><strong>Reading:</strong> {escape(interpretation)}</p>',
        unsafe_allow_html=True,
    )
    with st.expander("View chart data"):
        st.dataframe(accessible.round(4), width="stretch")

    st.markdown("### Strongest global model signals")
    st.caption(
        "Mean absolute SHAP values rank influence in the fitted model. They do not establish a causal effect in the Irish market."
    )
    ranking = [
        ("01", "Products held", "2.841"),
        ("02", "Months since switching", "1.028"),
        ("03", "Direct debits", "0.883"),
        ("04", "Tenure", "0.838"),
        ("05", "Savings goal", "0.529"),
    ]
    ranking_html = "".join(
        '<li><span class="evidence-ranking__rank">' + rank + "</span>"
        '<span class="evidence-ranking__name">' + escape(name) + "</span>"
        '<span class="evidence-ranking__value">' + value + "</span></li>"
        for rank, name, value in ranking
    )
    st.markdown(
        '<ol class="evidence-ranking" aria-label="Five strongest features by mean absolute SHAP value">'
        + ranking_html
        + "</ol>",
        unsafe_allow_html=True,
    )

    with st.expander("Inspection artifacts · SHAP beeswarm and bar views"):
        col_a, col_b = st.columns(2)
        beeswarm = assets_dir / "shap_summary_plot.png"
        bar = assets_dir / "shap_bar_plot.png"
        with col_a:
            st.markdown("**Feature interactions**")
            if beeswarm.exists():
                st.image(str(beeswarm), width="stretch")
                st.caption("Each dot is one holdout record. Position shows direction and magnitude of model contribution.")
            else:
                st.warning("Beeswarm artifact unavailable. Run models/train_model.py to regenerate it.")
        with col_b:
            st.markdown("**Mean absolute impact**")
            if bar.exists():
                st.image(str(bar), width="stretch")
                st.caption("Longer bars indicate larger average absolute contribution to raw model output.")
            else:
                st.warning("SHAP bar artifact unavailable. Run models/train_model.py to regenerate it.")

    st.markdown(
        '<section class="hard-boundary evidence-boundary" role="note">'
        '<span>Evidence boundary</span><strong>Model inspection is not legal or causal proof.</strong>'
        "<p>These scores describe performance on generated holdout records. SHAP explains the fitted model; "
        "it does not explain a real market, demonstrate fairness, or establish compliance with Article 86.</p>"
        "</section>",
        unsafe_allow_html=True,
    )
