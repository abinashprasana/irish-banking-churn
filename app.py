import os
import json
import warnings
from datetime import datetime, timezone
from html import escape
from pathlib import Path
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import shap
import dice_ml

from agent.tools import (
    Phase1SchemaError,
    load_phase1_runtime,
    predict_customer_churn_risk,
)
from lab_workspaces import render_data_limits, render_model_evidence
from decision_gate_ui import build_decision_stages, recommendation_state
from lab_ui.decision_instrument import render_decision_instrument
from case_review import (
    CaseAssessment,
    LabState,
    assessment_state_for_draft,
    build_phase1_customer,
    can_handoff_assessment,
    case_input_fingerprint,
    normalize_case_profile,
    risk_band_for_probability,
)

APP_DIR = Path(__file__).resolve().parent
BRAND_DIR = APP_DIR / "assets" / "brand"
BRAND_MARK_PATH = BRAND_DIR / "atlantic-ledger-mark.svg"
BRAND_FAVICON_PATH = BRAND_DIR / "atlantic-ledger-favicon-128.png"
PREMIUM_CSS_PATH = APP_DIR / "assets" / "lab-premium.css"


def _load_inline_brand_mark():
    """Load the generated, decorative Ledger Gate mark once."""
    return BRAND_MARK_PATH.read_text(encoding="utf-8")


BRAND_MARK_SVG = _load_inline_brand_mark()

SHAP_PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": "#071827",
    "axes.edgecolor": "#BDC7C9",
    "axes.labelcolor": "#071827",
    "axes.facecolor": "#F4F1E8",
    "figure.facecolor": "#F4F1E8",
    "savefig.facecolor": "#F4F1E8",
    "xtick.color": "#52636E",
    "ytick.color": "#071827",
    "grid.color": "#D7E1E6",
}

st.set_page_config(
    page_title="Atlantic Ledger · Interactive Lab",
    page_icon=str(BRAND_FAVICON_PATH),
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
:root {
    --ink: #102A43;
    --atlantic: #245B78;
    --atlantic-bright: #4EA2C6;
    --deep-ocean: #071827;
    --deep-ocean-soft: #0D2335;
    --cloud: #E4ECEE;
    --paper: #F7FAFA;
    --approval: #147D64;
    --approval-soft: #DFF3EC;
    --block: #A33A32;
    --block-soft: #F8E6E3;
    --line: #C8D6DA;
    --line-soft: #DCE6E8;
    --muted: #5C6F7E;
    --scene-mint: #68D5B3;
    --scene-blue: #8EC7DD;
    --radius-shell: 24px;
    --radius-panel: 18px;
    --radius-control: 10px;
    --space-unit: 0.5rem;
    --shadow-soft: 0 18px 45px rgba(16, 42, 67, 0.08);
    --shadow-lift: 0 22px 55px rgba(16, 42, 67, 0.13);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "IBM Plex Sans", Arial, sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 48rem 36rem at -6% -2%, rgba(72, 154, 153, 0.22), transparent 70%),
        radial-gradient(ellipse 52rem 40rem at 106% 14%, rgba(58, 112, 143, 0.2), transparent 72%),
        radial-gradient(ellipse 42rem 28rem at 58% 92%, rgba(103, 181, 166, 0.1), transparent 76%),
        linear-gradient(180deg, #E7EFF0 0%, #DAE6E8 48%, #E5EDEE 100%);
    position: relative;
}

[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container,
[data-testid="stMainBlockContainer"] {
    max-width: 1440px;
    margin: 0 auto;
    padding: 2.15rem 3rem 5rem !important;
    position: relative;
    width: 100%;
    z-index: 1;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--ink) !important;
    font-family: "IBM Plex Sans", Arial, sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.008em !important;
}

h2 {
    font-size: clamp(1.55rem, 2.4vw, 2rem) !important;
    letter-spacing: -0.028em !important;
    margin-top: 1.6rem !important;
    padding-top: 0.95rem;
    position: relative;
}

h2::before {
    background: var(--atlantic);
    border-radius: 999px;
    content: "";
    height: 2px;
    left: 0;
    position: absolute;
    top: 0;
    width: 32px;
}

h3 {
    font-size: 1.14rem !important;
}

p, li, .stMarkdown p {
    color: var(--ink);
    line-height: 1.6;
}

code, [data-testid="stMetricValue"], .ledger-value, .stat-value {
    font-family: "IBM Plex Mono", Consolas, monospace !important;
    font-variant-numeric: tabular-nums;
}

hr {
    border-color: var(--line) !important;
    margin: 1.5rem 0 !important;
}

.page-masthead {
    background:
        radial-gradient(ellipse 38% 70% at 84% 48%, rgba(78, 162, 198, 0.18), transparent 72%),
        radial-gradient(ellipse 28% 42% at 72% 100%, rgba(104, 213, 179, 0.07), transparent 75%),
        var(--deep-ocean);
    border: 0;
    border-radius: var(--radius-shell);
    background-clip: padding-box;
    box-shadow: 0 28px 70px rgba(7, 24, 39, 0.22), inset 0 0 0 1px rgba(191, 218, 231, 0.13);
    color: #FFFFFF;
    display: grid;
    grid-template-columns: minmax(0, 1.12fr) minmax(300px, 0.88fr);
    isolation: isolate;
    min-height: 300px;
    overflow: clip;
    position: relative;
}

.page-masthead::after {
    background: linear-gradient(90deg, #4EA2C6 0%, #68D5B3 46%, transparent 100%);
    content: "";
    height: 2px;
    left: 0;
    position: absolute;
    top: 0;
    transform-origin: left;
    width: 58%;
}

.hero-copy {
    align-self: center;
    padding: 2.35rem 1.4rem 2.35rem 2.6rem;
    position: relative;
    z-index: 2;
}

.hero-brand {
    align-items: center;
    display: flex;
    gap: 0.68rem;
    min-height: 34px;
}

.hero-brand-mark {
    align-items: center;
    color: #F4F1E8;
    display: flex;
    flex: 0 0 auto;
    height: 34px;
    justify-content: center;
    width: 34px;
}

.hero-brand-mark .brand-mark-svg {
    display: block;
    height: 100%;
    overflow: visible;
    width: 100%;
}

.hero-brand-mark .ledger-gate-part {
    fill: currentColor;
}

.hero-brand-name {
    color: #F4F1E8;
    font-family: "Source Serif 4", Georgia, serif;
    font-size: 0.9375rem;
    font-kerning: normal;
    font-optical-sizing: auto;
    font-weight: 600;
    letter-spacing: -0.012em;
    line-height: 1.08;
    white-space: nowrap;
}

.page-masthead h1 {
    color: #FFFFFF !important;
    font-size: clamp(1.85rem, calc(1.5rem + 1.7vw), 3.15rem);
    font-weight: 500 !important;
    letter-spacing: -0.045em;
    line-height: 1.02;
    margin: 0.95rem 0 0.85rem;
    max-width: 780px;
    padding: 0 !important;
}

.page-masthead h1 .hero-line,
.page-masthead h1 .hero-line > span,
.page-masthead h1 .hero-accent {
    display: block;
    text-wrap: balance;
}

.page-masthead h1 .hero-line {
    overflow: clip;
    padding-bottom: 0.08em;
}

.page-masthead h1 .hero-line + .hero-line {
    margin-top: -0.08em;
}

.page-masthead h1 .hero-line > span {
    animation: headlineReveal 640ms cubic-bezier(0.16, 1, 0.3, 1) both;
    will-change: transform, opacity;
}

.page-masthead h1 .hero-line:nth-child(1) > span {
    animation-delay: 80ms;
}

.page-masthead h1 .hero-line:nth-child(2) > span {
    animation-delay: 155ms;
}

.page-masthead h1 [data-testid="stHeaderActionElements"] {
    display: none !important;
}

.page-masthead h1 .hero-accent {
    color: #8EC7DD;
}

.page-masthead p {
    color: #C7D8E2;
    font-size: clamp(0.98rem, 1.4vw, 1.08rem);
    line-height: 1.7;
    margin: 0;
    max-width: 650px;
}

.hero-facts {
    border-top: 1px solid rgba(207, 231, 241, 0.16);
    display: grid;
    gap: 0;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin-top: 1.3rem;
    max-width: 670px;
    padding-top: 0.85rem;
}

.hero-fact {
    animation: factReveal 440ms cubic-bezier(0.16, 1, 0.3, 1) both;
    min-width: 0;
    padding-right: 1rem;
}

.hero-fact:nth-child(1) { animation-delay: 300ms; }
.hero-fact:nth-child(2) { animation-delay: 355ms; }
.hero-fact:nth-child(3) { animation-delay: 410ms; }

.hero-fact + .hero-fact {
    border-left: 1px solid rgba(207, 231, 241, 0.16);
    padding-left: 1rem;
}

.hero-fact strong {
    color: #FFFFFF;
    display: block;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.92rem;
    font-weight: 600;
    margin-bottom: 0.25rem;
}

.hero-fact span {
    color: #91AAB8;
    display: block;
    font-size: 0.72rem;
    line-height: 1.35;
}

.hero-scene {
    align-items: center;
    display: flex;
    justify-content: center;
    min-height: 300px;
    overflow: clip;
    padding: 1.6rem;
    perspective: 1100px;
    perspective-origin: 48% 48%;
    position: relative;
    transform-style: preserve-3d;
    transform: scale(0.82);
}

.hero-scene::before {
    background-image:
        linear-gradient(rgba(207, 231, 241, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(207, 231, 241, 0.05) 1px, transparent 1px);
    background-size: 42px 42px;
    content: "";
    inset: 0;
    mask-image: radial-gradient(ellipse at center, black 28%, rgba(0, 0, 0, 0.72) 68%, transparent 100%);
    -webkit-mask-image: radial-gradient(ellipse at center, black 28%, rgba(0, 0, 0, 0.72) 68%, transparent 100%);
    opacity: 0.78;
    pointer-events: none;
    position: absolute;
    transform: rotateX(58deg) translate3d(0, 7%, -90px) scale(1.22);
    transform-origin: center;
    z-index: 0;
}

.hero-scene::after {
    background:
        radial-gradient(ellipse 52% 22% at 52% 67%, rgba(104, 213, 179, 0.16), transparent 72%),
        radial-gradient(circle at 62% 48%, rgba(78, 162, 198, 0.12), transparent 48%);
    content: "";
    inset: 8%;
    pointer-events: none;
    position: absolute;
    transform: translateZ(-55px);
    z-index: 0;
}

.scene-frame {
    aspect-ratio: 1;
    background:
        radial-gradient(circle at 68% 63%, rgba(78, 162, 198, 0.08), transparent 34%),
        linear-gradient(145deg, rgba(191, 218, 231, 0.04), transparent 48%);
    border: 1px solid rgba(191, 218, 231, 0.18);
    border-radius: 28px;
    box-shadow:
        0 30px 65px rgba(0, 0, 0, 0.18),
        inset 0 1px 0 rgba(191, 218, 231, 0.05);
    max-width: 410px;
    position: relative;
    transform: rotateX(3deg) rotateY(-3deg) rotateZ(-3deg);
    transform-style: preserve-3d;
    transition: transform 420ms cubic-bezier(0.16, 1, 0.3, 1), border-color 260ms ease, box-shadow 260ms ease;
    will-change: transform;
    width: 100%;
    z-index: 1;
}

.scene-frame::before {
    border: 1px dashed rgba(78, 162, 198, 0.28);
    border-radius: 50%;
    content: "";
    inset: 25%;
    pointer-events: none;
    position: absolute;
    transform: translateZ(38px) rotateX(58deg) scale(1.1);
    z-index: 0;
}

.scene-frame::after {
    border: 1px solid rgba(104, 213, 179, 0.16);
    border-radius: 50%;
    box-shadow: 0 0 34px rgba(78, 162, 198, 0.08);
    content: "";
    inset: 15% 12%;
    pointer-events: none;
    position: absolute;
    transform: translateZ(-28px) rotateX(68deg) rotateZ(-10deg);
    z-index: 0;
}

@media (hover: hover) and (pointer: fine) {
    .scene-frame:hover {
        border-color: rgba(191, 218, 231, 0.28);
        box-shadow:
            0 38px 72px rgba(0, 0, 0, 0.22),
            inset 0 1px 0 rgba(191, 218, 231, 0.08);
        transform: rotateX(1deg) rotateY(4deg) rotateZ(-1deg) translateY(-3px);
    }
}

.scene-axis {
    background: linear-gradient(90deg, transparent, rgba(142, 199, 221, 0.4), transparent);
    height: 1px;
    left: 5%;
    position: absolute;
    right: 5%;
    top: 50%;
    display: none;
}

.scene-axis.vertical {
    left: 50%;
    top: 5%;
    transform: rotate(90deg);
    transform-origin: left;
    width: 90%;
}

.scene-core {
    align-items: center;
    background:
        radial-gradient(circle at 35% 30%, rgba(78, 162, 198, 0.15), transparent 46%),
        rgba(13, 35, 53, 0.94);
    border: 1px solid rgba(142, 199, 221, 0.55);
    border-radius: 50%;
    box-shadow: 0 0 0 14px rgba(78, 162, 198, 0.06), 0 0 38px rgba(78, 162, 198, 0.16);
    display: flex;
    flex-direction: column;
    height: 128px;
    justify-content: center;
    left: 61%;
    position: absolute;
    text-align: center;
    top: 52%;
    transform: translate(-50%, -50%) translateZ(82px);
    transform-style: preserve-3d;
    width: 128px;
    z-index: 4;
}

.scene-core-brand {
    color: #F4F1E8;
    inset: 8px;
    opacity: 0.09;
    pointer-events: none;
    position: absolute;
    transform: translateZ(-1px);
}

.scene-core-brand .brand-mark-svg {
    display: block;
    height: 100%;
    width: 100%;
}

.scene-core-brand .brand-mark-svg path {
    fill: currentColor;
}

.scene-core small {
    color: #8EC7DD;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}

.scene-core strong {
    color: #FFFFFF;
    font-size: 0.96rem;
    font-weight: 500;
    margin-top: 0.25rem;
    position: relative;
    z-index: 1;
}

.scene-node {
    align-items: center;
    backdrop-filter: blur(9px);
    background: rgba(7, 24, 39, 0.9);
    border: 1px solid rgba(191, 218, 231, 0.24);
    border-radius: 10px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    color: #D9E7ED;
    display: flex;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.6rem;
    gap: 0.45rem;
    letter-spacing: 0.08em;
    box-sizing: border-box;
    padding: 0.68rem 0.78rem;
    position: absolute;
    text-transform: uppercase;
    white-space: nowrap;
    z-index: 5;
}

.scene-node::before {
    background: #68D5B3;
    border-radius: 1px;
    content: "";
    display: block;
    flex: 0 0 auto;
    height: 2px;
    width: 10px;
}

.scene-node.input {
    left: 7%;
    top: 14%;
    transform: translateZ(58px);
}

.scene-node.risk {
    right: 6%;
    top: 17%;
    transform: translateZ(72px);
}

.scene-node.reason {
    bottom: 12%;
    left: 5%;
    transform: translateZ(66px);
}

.scene-node.policy {
    bottom: 12%;
    right: 7%;
    transform: translateZ(78px);
}

.scene-sweep {
    border: 1px solid transparent;
    border-radius: 50%;
    border-top-color: rgba(142, 199, 221, 0.82);
    inset: 17%;
    opacity: 0.75;
    pointer-events: none;
    position: absolute;
    transform: translateZ(44px) rotateX(60deg) rotateZ(0deg);
    transform-style: preserve-3d;
    z-index: 2;
}

.scene-flow {
    inset: 0;
    overflow: visible;
    pointer-events: none;
    position: absolute;
    transform: translateZ(26px);
    z-index: 2;
}

.scene-flow path {
    fill: none;
    vector-effect: non-scaling-stroke;
}

.scene-flow-base {
    stroke: rgba(191, 218, 231, 0.2);
    stroke-width: 1;
}

.scene-flow-signal {
    animation: flowSignal 5.8s linear infinite;
    stroke: #68D5B3;
    stroke-dasharray: 3 46;
    stroke-linecap: round;
    stroke-width: 1.7;
}

.scene-orbit {
    fill: none;
    stroke-linecap: round;
    transform-box: fill-box;
    transform-origin: center;
    vector-effect: non-scaling-stroke;
}

.scene-orbit.outer {
    stroke: rgba(142, 199, 221, 0.32);
    stroke-dasharray: 155 48;
    stroke-width: 1.15;
    transform: rotate(-22deg);
}

.scene-orbit.inner {
    stroke: rgba(104, 213, 179, 0.28);
    stroke-dasharray: 56 34;
    stroke-width: 1;
    transform: rotate(31deg);
}

.scene-caption {
    bottom: 0.35rem;
    color: #8AA3B1;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.55rem;
    left: 1rem;
    letter-spacing: 0.08em;
    position: absolute;
    text-transform: uppercase;
    transform: translateZ(46px);
    z-index: 5;
}

.eyebrow {
    color: var(--atlantic) !important;
    font-size: 0.72rem !important;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.section-note {
    color: var(--muted);
    max-width: 820px;
    margin-top: -0.35rem;
}

.stTabs [data-baseweb="tab-list"] {
    backdrop-filter: blur(20px) saturate(160%);
    background:
        linear-gradient(180deg, rgba(249, 252, 251, 0.86), rgba(237, 244, 244, 0.8));
    background-clip: padding-box;
    border: 1px solid rgba(188, 207, 211, 0.9);
    border-radius: calc(var(--radius-control) + 4px);
    box-shadow:
        0 16px 38px rgba(16, 42, 67, 0.12),
        inset 0 1px 0 rgba(255, 255, 255, 0.78);
    gap: 0.35rem;
    margin: -1.05rem 1.35rem 0;
    padding: 0.42rem;
    overflow-x: auto;
    position: sticky;
    top: 0.7rem;
    scrollbar-width: none;
    scroll-snap-type: x proximity;
    transition: box-shadow 220ms ease, background-color 220ms ease;
    z-index: 5;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    display: none;
}

[data-testid="stTabsScrollLeft"],
[data-testid="stTabsScrollRight"] {
    align-items: center;
    background: rgba(249, 252, 251, 0.9) !important;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(188, 207, 211, 0.9) !important;
    border-radius: 999px !important;
    box-shadow: 0 10px 22px rgba(16, 42, 67, 0.12);
    color: var(--atlantic) !important;
    display: flex !important;
    height: 2.1rem;
    justify-content: center;
    margin-top: 0.55rem;
    transition: background-color 160ms ease, transform 160ms ease, box-shadow 160ms ease;
    width: 2.1rem;
}

[data-testid="stTabsScrollLeft"]:hover,
[data-testid="stTabsScrollRight"]:hover {
    background: #FFFFFF !important;
    box-shadow: 0 12px 26px rgba(16, 42, 67, 0.16);
    transform: translateY(-1px);
}

[data-testid="stTabsScrollLeft"]:active,
[data-testid="stTabsScrollRight"]:active {
    transform: translateY(0) scale(0.94);
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 0 !important;
    border-radius: var(--radius-control) !important;
    color: var(--muted) !important;
    flex: 0 0 auto;
    font-size: 0.84rem;
    font-weight: 500;
    overflow: hidden;
    padding: 0.78rem 1rem;
    position: relative;
    scroll-snap-align: start;
    transition: background-color 180ms ease, color 180ms ease, box-shadow 180ms ease, transform 180ms ease;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #EDF4F7 !important;
    color: var(--atlantic) !important;
    transform: translateY(-1px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(145deg, #102A43, #0C2134) !important;
    box-shadow: 0 7px 16px rgba(16, 42, 67, 0.18);
    color: var(--ink) !important;
    transform: translateY(0);
}

.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #68D5B3, #4EA2C6) !important;
    border-radius: 999px;
    height: 2px !important;
    transition:
        transform 240ms cubic-bezier(0, 0, 0.38, 0.9),
        right 240ms cubic-bezier(0, 0, 0.38, 0.9),
        width 240ms cubic-bezier(0, 0, 0.38, 0.9) !important;
    z-index: 7;
}

.stTabs [data-baseweb="tab-border"] {
    background: rgba(36, 91, 120, 0.12) !important;
    height: 1px !important;
}

.stTabs [data-baseweb="tab-panel"] {
    animation: panelEnter 360ms cubic-bezier(0.16, 1, 0.3, 1) both;
    padding-top: 1.4rem;
}

.stat-band {
    display: grid;
    grid-template-columns: repeat(var(--stat-count, 3), minmax(0, 1fr));
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: var(--radius-panel);
    box-shadow: var(--shadow-soft);
    margin: 1.2rem 0 1.5rem;
    overflow: hidden;
}

.stat-item {
    animation: metricEnter 480ms cubic-bezier(0.16, 1, 0.3, 1) both;
    min-height: 150px;
    padding: 1.4rem 1.45rem 1.25rem;
    position: relative;
    transition: background-color 180ms ease, transform 180ms ease, box-shadow 180ms ease;
}

.stat-item:nth-child(1) { animation-delay: 60ms; }
.stat-item:nth-child(2) { animation-delay: 110ms; }
.stat-item:nth-child(3) { animation-delay: 160ms; }
.stat-item:nth-child(4) { animation-delay: 210ms; }
.stat-item:nth-child(5) { animation-delay: 260ms; }

.stat-item::before {
    background: var(--atlantic);
    content: "";
    height: 3px;
    left: 1.45rem;
    position: absolute;
    top: 0;
    width: 38px;
}

.stat-item:hover {
    background: rgba(248, 251, 252, 0.98);
    box-shadow: inset 0 0 0 1px rgba(78, 162, 198, 0.08);
    transform: translateY(-2px);
}

.stat-item + .stat-item {
    border-left: 1px solid var(--line);
}

.stat-item.approval {
    box-shadow: none;
}

.stat-item.approval::before {
    background: var(--approval);
}

.stat-item.block {
    box-shadow: none;
}

.stat-item.block::before {
    background: var(--block);
}

.stat-label {
    color: var(--muted);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.stat-value {
    color: var(--ink);
    font-size: clamp(1.95rem, 3vw, 2.7rem);
    font-weight: 600;
    line-height: 1.15;
    margin: 0.28rem 0 0.36rem;
}

.stat-note {
    color: var(--muted);
    font-size: 0.82rem;
    line-height: 1.45;
}

.feature-panel {
    background: var(--paper);
    border: 1px solid rgba(173, 190, 199, 0.62);
    border-radius: var(--radius-panel);
    box-shadow: 0 12px 30px rgba(16, 42, 67, 0.065);
    min-height: 220px;
    overflow: hidden;
    padding: 1.35rem 1.35rem 1.2rem;
    position: relative;
    transition: border-color 180ms ease, box-shadow 180ms ease, transform 180ms cubic-bezier(0.16, 1, 0.3, 1);
}

.feature-panel::before {
    background: linear-gradient(90deg, transparent, rgba(78, 162, 198, 0.55), rgba(104, 213, 179, 0.5), transparent);
    content: "";
    height: 1px;
    left: 10%;
    opacity: 0;
    position: absolute;
    right: 10%;
    top: 0;
    transform: scaleX(0.45);
    transition: opacity 180ms ease, transform 260ms cubic-bezier(0.16, 1, 0.3, 1);
}

.feature-panel::after {
    border-right: 1px solid rgba(36, 91, 120, 0.3);
    border-top: 1px solid rgba(36, 91, 120, 0.3);
    content: "";
    height: 18px;
    position: absolute;
    right: 11px;
    top: 11px;
    transition: border-color 220ms ease, height 220ms ease, width 220ms ease;
    width: 18px;
}

.feature-panel:hover {
    border-color: #AFC4D0;
    box-shadow: 0 16px 36px rgba(16, 42, 67, 0.09);
    transform: translateY(-3px);
}

.feature-panel:hover::before {
    opacity: 1;
    transform: scaleX(1);
}

.feature-panel:hover::after {
    border-color: var(--atlantic);
}

.feature-index {
    color: #91A8B5;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.64rem;
    letter-spacing: 0.08em;
}

.feature-visual {
    align-items: flex-end;
    display: flex;
    gap: 5px;
    height: 44px;
    margin-bottom: 1.15rem;
}

.feature-visual span {
    background: #BFD8E4;
    border-radius: 2px 2px 0 0;
    display: block;
    width: 7px;
}

.feature-visual span:nth-child(1) { height: 18px; }
.feature-visual span:nth-child(2) { height: 32px; }
.feature-visual span:nth-child(3) { height: 24px; }
.feature-visual span:nth-child(4) { background: var(--atlantic); height: 42px; }
.feature-visual span:nth-child(5) { height: 29px; }

.feature-panel.explain .feature-visual {
    align-items: center;
    gap: 0;
}

.feature-panel.explain .feature-visual span {
    background: var(--line);
    border-radius: 999px;
    height: 2px;
    position: relative;
    width: 36px;
}

.feature-panel.explain .feature-visual span::after {
    background: var(--atlantic);
    border: 3px solid #D9EAF1;
    border-radius: 50%;
    content: "";
    height: 10px;
    left: 50%;
    position: absolute;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 10px;
}

.feature-panel.counter .feature-visual {
    align-items: center;
}

.feature-panel.counter .feature-visual span {
    background: transparent;
    border: 1px solid #AFC4D0;
    border-radius: 50%;
    height: 30px;
    position: relative;
    width: 30px;
}

.feature-panel.counter .feature-visual span + span {
    margin-left: -10px;
}

.feature-panel.counter .feature-visual span:last-child {
    border-color: var(--approval);
}

.feature-panel.counter .feature-visual span:nth-child(n+4) {
    display: none;
}

.feature-panel h4 {
    font-size: 1.05rem;
    margin: 0 0 0.45rem;
}

.feature-panel p {
    color: var(--muted);
    font-size: 0.9rem;
}

.system-story {
    background:
        radial-gradient(ellipse 34% 24% at 96% 8%, rgba(78, 162, 198, 0.1), transparent 74%),
        rgba(245, 249, 248, 0.82);
    background-clip: padding-box;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(173, 190, 199, 0.55);
    border-radius: var(--radius-shell);
    box-shadow: var(--shadow-soft);
    display: grid;
    gap: clamp(2.2rem, 5vw, 5.5rem);
    grid-template-columns: minmax(300px, 0.82fr) minmax(0, 1.18fr);
    margin: 2.7rem 0 1.5rem;
    overflow: clip;
    padding: clamp(1.25rem, 3vw, 2.25rem);
    position: relative;
    view-timeline-axis: block;
    view-timeline-name: --system-story;
}

.system-story::before {
    display: none;
}

.story-map {
    align-self: start;
    background:
        radial-gradient(ellipse 72% 54% at 72% 62%, rgba(78, 162, 198, 0.16), transparent 72%),
        linear-gradient(160deg, #071827 0%, #0B2132 100%);
    border-radius: var(--radius-panel);
    box-shadow: 0 20px 45px rgba(7, 24, 39, 0.18);
    min-height: 620px;
    overflow: hidden;
    padding: 1.5rem;
    perspective: 900px;
    position: sticky;
    top: 5.5rem;
}

.story-map::after {
    background: radial-gradient(ellipse at center, rgba(104, 213, 179, 0.12), transparent 70%);
    bottom: -8%;
    content: "";
    height: 54%;
    pointer-events: none;
    position: absolute;
    right: -15%;
    transform: rotate(-12deg);
    width: 78%;
    z-index: 0;
}

.story-map-copy {
    max-width: 420px;
    position: relative;
    z-index: 2;
}

.story-map-copy .eyebrow {
    color: #8EC7DD !important;
}

.story-map-copy h3 {
    color: #FFFFFF !important;
    font-size: clamp(1.7rem, 3vw, 2.45rem) !important;
    line-height: 1.06;
    margin: 0.6rem 0 0.85rem;
}

.story-map-copy p {
    color: #BFD1DB;
    font-size: 0.92rem;
    margin: 0;
}

.story-map-graphic {
    bottom: 0;
    height: 430px;
    left: 1.5rem;
    max-width: 420px;
    position: absolute;
    transform: translateZ(22px);
    width: calc(100% - 3rem);
    z-index: 1;
}

.story-depth-line {
    fill: none;
    stroke-linecap: round;
    vector-effect: non-scaling-stroke;
}

.story-depth-line.one {
    stroke: rgba(142, 199, 221, 0.14);
    stroke-width: 1;
}

.story-depth-line.two {
    stroke: rgba(104, 213, 179, 0.12);
    stroke-dasharray: 82 24;
    stroke-width: 1;
}

.story-track,
.story-progress {
    fill: none;
    stroke-linecap: round;
    stroke-linejoin: round;
    vector-effect: non-scaling-stroke;
}

.story-track {
    stroke: rgba(191, 218, 231, 0.28);
    stroke-width: 1.5;
}

.story-progress {
    filter: drop-shadow(0 0 5px rgba(104, 213, 179, 0.38));
    stroke: #68D5B3;
    stroke-width: 2.8;
}

.story-node {
    fill: var(--deep-ocean);
    stroke: #8EC7DD;
    stroke-width: 1.5;
    transform-box: fill-box;
    transform-origin: center;
}

.story-node.core {
    fill: none;
    stroke: #68D5B3;
    stroke-linecap: square;
    stroke-width: 2.2;
}

.story-node-check {
    fill: none;
    stroke: #F7FAFA;
    stroke-linecap: square;
    stroke-linejoin: miter;
    stroke-width: 2.2;
}

.story-node-label {
    fill: #BFD1DB;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
}

.story-steps {
    display: grid;
    gap: clamp(2rem, 7vh, 4.5rem);
    padding: clamp(2rem, 6vh, 4.5rem) 0;
    position: relative;
}

.story-step {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid rgba(173, 190, 199, 0.58);
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(16, 42, 67, 0.07);
    min-height: 155px;
    padding: 1.3rem 1.4rem 1.35rem 4.4rem;
    position: relative;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}

.story-step:hover {
    border-color: rgba(78, 162, 198, 0.45);
    box-shadow: 0 18px 42px rgba(16, 42, 67, 0.1);
}

.story-step-number {
    align-items: center;
    background: #E6F0F4;
    border: 1px solid #C5D8E1;
    border-radius: 50%;
    color: var(--atlantic);
    display: flex;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.66rem;
    font-weight: 600;
    height: 34px;
    justify-content: center;
    left: 1.25rem;
    position: absolute;
    top: 1.25rem;
    width: 34px;
}

.story-step h4 {
    font-size: 1rem;
    margin: 0 0 0.42rem;
}

.story-step p {
    color: var(--muted);
    font-size: 0.88rem;
    line-height: 1.62;
    margin: 0;
}

.story-proof {
    border-top: 1px solid var(--line-soft);
    color: var(--atlantic);
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.62rem;
    letter-spacing: 0.04em;
    margin-top: 0.85rem;
    padding-top: 0.72rem;
}

.risk-panel {
    animation: resultEnter 400ms cubic-bezier(0.16, 1, 0.3, 1) both;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: var(--radius-panel);
    box-shadow: var(--shadow-soft);
    overflow: hidden;
    padding: 1.45rem 1.55rem;
    margin: 0.8rem 0;
    position: relative;
}

.risk-panel::before {
    background: var(--atlantic);
    content: "";
    inset: 0 auto 0 0;
    position: absolute;
    width: 5px;
}

.risk-panel.low {
    border-left-color: var(--line);
    box-shadow: 0 18px 42px rgba(20, 125, 100, 0.1);
}

.risk-panel.low::before {
    background: var(--approval);
}

.risk-panel.high {
    border-left-color: var(--line);
    box-shadow: 0 18px 42px rgba(163, 58, 50, 0.1);
}

.risk-panel.high::before {
    background: var(--block);
}

.risk-topline {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
}

.risk-value {
    animation: numeralPop 380ms cubic-bezier(0.16, 1, 0.3, 1) 200ms both;
    color: var(--ink);
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: clamp(2.35rem, 5vw, 3.3rem);
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-top: 0.35rem;
}

.status-label {
    border: 1px solid currentColor;
    border-radius: 999px;
    color: var(--atlantic);
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    padding: 0.38rem 0.55rem;
    text-transform: uppercase;
}

.status-label.low {
    color: var(--approval);
}

.status-label.high {
    color: var(--block);
}

.risk-panel p {
    color: var(--muted);
    margin: 0.7rem 0 0;
}

.prob-bar {
    margin: 0.95rem 0.1rem 0.15rem;
}

.prob-bar-track {
    background: var(--line-soft);
    border-radius: 999px;
    height: 10px;
    overflow: visible;
    position: relative;
}

.prob-bar-fill {
    animation: fillBar 650ms cubic-bezier(0.16, 1, 0.3, 1) 150ms both;
    background: var(--atlantic);
    border-radius: 999px;
    height: 100%;
    transform: scaleX(0);
    transform-origin: left;
    width: var(--target-width);
}

.prob-bar-fill.low {
    background: var(--approval);
}

.prob-bar-fill.high {
    background: var(--block);
}

.prob-bar-tick {
    background: rgba(16, 42, 67, 0.28);
    border-radius: 1px;
    height: 16px;
    position: absolute;
    top: -3px;
    transform: translateX(-50%);
    width: 2px;
}

.prob-bar-labels {
    height: 1rem;
    margin-top: 0.3rem;
    position: relative;
}

.prob-bar-labels span {
    color: var(--muted);
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.68rem;
    position: absolute;
    transform: translateX(-50%);
}

@keyframes fillBar {
    from { transform: scaleX(0); }
    to { transform: scaleX(1); }
}

.empty-panel {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: var(--radius-panel);
    box-shadow: var(--shadow-soft);
    padding: 2.4rem 2rem;
    margin-top: 0.8rem;
}

.empty-panel strong {
    display: block;
    font-size: 1.05rem;
    margin-bottom: 0.45rem;
}

.empty-panel p {
    color: var(--muted);
    font-size: 0.9rem;
    margin: 0;
}

.policy-ledger {
    animation: resultEnter 400ms cubic-bezier(0.16, 1, 0.3, 1) both;
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: var(--radius-panel);
    box-shadow: var(--shadow-soft);
    margin: 1rem 0;
    overflow: hidden;
    position: relative;
}

.policy-ledger::after {
    background: radial-gradient(circle, rgba(20, 125, 100, 0.12), transparent 68%);
    content: "";
    height: 150px;
    pointer-events: none;
    position: absolute;
    right: -55px;
    top: -65px;
    width: 150px;
}

.policy-ledger::before {
    background: var(--approval);
    content: "";
    inset: 0 auto 0 0;
    position: absolute;
    width: 6px;
}

.policy-ledger.blocked {
    border-left-color: var(--line);
}

.policy-ledger.blocked::before {
    background: var(--block);
}

.policy-ledger.blocked::after {
    background: radial-gradient(circle, rgba(163, 58, 50, 0.12), transparent 68%);
}

.policy-ledger.review {
    border-left-color: var(--line);
}

.policy-ledger.review::before {
    background: var(--atlantic);
}

.policy-ledger.review::after {
    background: radial-gradient(circle, rgba(36, 91, 120, 0.12), transparent 68%);
}

.ledger-verdict {
    border-bottom: 1px solid var(--line);
    padding: 1.35rem 1.35rem 1.5rem;
}

.ledger-title {
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.ledger-state {
    align-items: center;
    animation: numeralPop 380ms cubic-bezier(0.16, 1, 0.3, 1) 200ms both;
    color: var(--approval);
    display: flex;
    font-family: "IBM Plex Sans", Arial, sans-serif;
    font-size: clamp(1.7rem, 3.4vw, 2.35rem);
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-top: 0.4rem;
}

.ledger-state::before {
    background: currentColor;
    border-radius: 999px;
    content: "";
    flex: 0 0 auto;
    height: 11px;
    margin-right: 0.65rem;
    width: 11px;
}

.policy-ledger.blocked .ledger-state {
    color: var(--block);
}

.policy-ledger.review .ledger-state {
    color: var(--atlantic);
}

@keyframes numeralPop {
    from {
        opacity: 0;
        transform: scale(0.94);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.ledger-grid {
    display: grid;
    grid-template-columns: 1.3fr 2fr 0.75fr;
}

.ledger-cell {
    padding: 1rem 1.2rem 1.15rem;
}

.ledger-cell + .ledger-cell {
    border-left: 1px solid var(--line);
}

.ledger-label {
    color: var(--muted);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.34rem;
    text-transform: uppercase;
}

.ledger-value {
    color: var(--ink);
    font-size: 0.9rem;
    overflow-wrap: anywhere;
}

.ledger-copy {
    color: var(--ink);
    font-size: 0.88rem;
    line-height: 1.55;
}

.ledger-footer {
    border-top: 1px solid var(--line);
    color: var(--muted);
    font-size: 0.8rem;
    padding: 0.8rem 1.2rem;
}


.stButton > button {
    border-radius: var(--radius-control) !important;
    font-family: "IBM Plex Sans", Arial, sans-serif;
    font-weight: 600 !important;
    min-height: 2.75rem;
    overflow: hidden;
    position: relative;
    transition: box-shadow 180ms ease, transform 180ms ease, background-color 180ms ease;
}

.stButton > button[kind="primary"] {
    background: var(--atlantic) !important;
    border: 1px solid var(--atlantic) !important;
    color: white !important;
    box-shadow: 0 10px 24px rgba(36, 91, 120, 0.2) !important;
}

.stButton > button[kind="primary"]:hover {
    background: var(--ink) !important;
    border-color: var(--ink) !important;
    box-shadow: 0 14px 30px rgba(16, 42, 67, 0.25) !important;
    transform: translateY(-2px);
}

.stButton > button:active {
    transform: translateY(0) scale(0.985) !important;
}

.stButton > button:focus-visible,
.stTabs [data-baseweb="tab"]:focus-visible,
input:focus-visible,
button:focus-visible {
    outline: 3px solid rgba(36, 91, 120, 0.35) !important;
    outline-offset: 2px;
}

[data-testid="metric-container"] {
    animation: metricEnter 440ms cubic-bezier(0.16, 1, 0.3, 1) both;
    background: transparent !important;
    border-left: 2px solid var(--atlantic);
    padding: 0.25rem 0.8rem;
}

[data-testid="stMetricValue"] {
    color: var(--ink) !important;
    font-size: 1.6rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricLabel"],
.stCaption,
[data-testid="stCaptionContainer"] {
    color: var(--muted) !important;
}

[data-testid="stAlertContainer"] {
    background: var(--paper) !important;
    border: 1px solid var(--line);
    border-left: 4px solid var(--atlantic);
    border-radius: 12px;
    color: var(--ink) !important;
    box-shadow: 0 8px 24px rgba(16, 42, 67, 0.045);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
    border-left-color: var(--approval);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
    border-left-color: var(--block);
}

[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
    border-left-color: var(--atlantic);
}

[data-testid="stAlertContainer"] p {
    color: var(--ink) !important;
}

[data-testid="stExpander"] {
    background: var(--paper);
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    overflow: hidden;
    transition: border-color 180ms ease, box-shadow 180ms ease;
}

[data-testid="stExpander"]:hover {
    border-color: #AFC4D0 !important;
    box-shadow: 0 9px 22px rgba(16, 42, 67, 0.055);
}

[data-testid="stExpander"] summary {
    color: var(--ink) !important;
    font-weight: 500;
}

[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stImage"] img,
.stPlotlyChart {
    border: 1px solid var(--line);
    border-radius: 14px !important;
    box-shadow: 0 12px 30px rgba(16, 42, 67, 0.055);
    overflow: hidden;
}

.stPlotlyChart {
    background: rgba(248, 251, 250, 0.72);
    transition: border-color 150ms ease, box-shadow 150ms ease;
}

@media (hover: hover) {
    .stPlotlyChart:hover {
        border-color: rgba(36, 91, 120, 0.38);
        box-shadow: 0 14px 32px rgba(16, 42, 67, 0.08);
    }
}

[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {
    background: var(--paper) !important;
    border-color: var(--line) !important;
    border-radius: 9px !important;
}

[data-testid="stProgress"] > div {
    border-radius: 999px;
    overflow: hidden;
}

[data-testid="stSlider"] [role="slider"] {
    background-color: var(--atlantic) !important;
    box-shadow: 0 0 0 4px rgba(36, 91, 120, 0.14) !important;
    transition: box-shadow 160ms ease, transform 120ms ease !important;
}

[data-testid="stSlider"] [role="slider"]:hover {
    box-shadow: 0 0 0 7px rgba(36, 91, 120, 0.16) !important;
    transition-duration: 0ms !important;
}

[data-testid="stSlider"] [role="slider"]:active {
    transform: scale(1.12);
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div:nth-child(2) {
    background: var(--atlantic) !important;
}

[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
    background: var(--line) !important;
}

[data-testid="stSlider"] [data-testid="stTickBar"] {
    display: none;
}

[data-testid="stCheckbox"] label span[data-testid="stMarkdownContainer"] {
    color: var(--ink);
}

[data-testid="stCheckbox"] label:has(input:checked) span:first-child {
    background-color: var(--atlantic) !important;
    border-color: var(--atlantic) !important;
}

[data-testid="stCheckbox"] label span:first-child {
    border-radius: 6px !important;
    transition: background-color 140ms ease, border-color 140ms ease, transform 120ms ease !important;
}

[data-testid="stCheckbox"] label:active span:first-child {
    transform: scale(0.92);
}

@keyframes heroRise {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes headlineReveal {
    from {
        opacity: 0;
        transform: translateY(108%);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes factReveal {
    from {
        opacity: 0;
        transform: translateY(9px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes panelEnter {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes metricEnter {
    from {
        opacity: 0;
        transform: translateY(10px) scale(0.985);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes resultEnter {
    from {
        opacity: 0;
        transform: translateY(12px) scale(0.99);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes storyStepEnter {
    from {
        opacity: 0.52;
        transform: translateY(16px) scale(0.985);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes flowSignal {
    to {
        stroke-dashoffset: -196;
    }
}

@keyframes sceneOrbit {
    from {
        transform: translateZ(44px) rotateX(60deg) rotateZ(0deg);
    }
    to {
        transform: translateZ(44px) rotateX(60deg) rotateZ(360deg);
    }
}

@keyframes sceneOrbitFlat {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
}

@keyframes coreSignal {
    0% {
        border-color: rgba(142, 199, 221, 0.35);
        box-shadow: 0 0 0 8px rgba(78, 162, 198, 0.025), 0 0 20px rgba(78, 162, 198, 0.08);
    }
    68% {
        border-color: rgba(142, 199, 221, 0.7);
        box-shadow: 0 0 0 17px rgba(78, 162, 198, 0.075), 0 0 44px rgba(78, 162, 198, 0.19);
    }
    100% {
        border-color: rgba(142, 199, 221, 0.55);
        box-shadow: 0 0 0 14px rgba(78, 162, 198, 0.06), 0 0 38px rgba(78, 162, 198, 0.16);
    }
}

@keyframes drawStory {
    from {
        stroke-dashoffset: 940;
    }
    to {
        stroke-dashoffset: 0;
    }
}

@keyframes wakeStoryNode {
    from {
        opacity: 0.35;
        transform: scale(0.78);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

.hero-copy > .hero-brand,
.hero-copy > p,
.hero-copy > .hero-facts {
    animation: heroRise 420ms cubic-bezier(0, 0, 0.38, 0.9) both;
}

.hero-copy p {
    animation-delay: 210ms;
}

.hero-facts {
    animation-delay: 270ms;
}

.scene-sweep {
    animation: sceneOrbit 22s linear infinite;
}

.scene-core {
    animation: coreSignal 900ms cubic-bezier(0.16, 1, 0.3, 1) 420ms both;
}

@supports (animation-timeline: view()) {
    .story-progress {
        animation: drawStory linear both;
        animation-range: entry 8% exit 82%;
        animation-timeline: --system-story;
        stroke-dasharray: 940;
    }

    .story-node {
        animation: wakeStoryNode linear both;
        animation-timeline: --system-story;
    }

    .story-node.n1 { animation-range: entry 7% entry 19%; }
    .story-node.n2 { animation-range: entry 20% entry 34%; }
    .story-node.n3 { animation-range: entry 35% entry 49%; }
    .story-node.n4 { animation-range: entry 50% entry 64%; }
    .story-node.n5 { animation-range: entry 65% entry 79%; }

    .story-step {
        animation: storyStepEnter linear both;
        animation-range: entry 5% entry 36%;
        animation-timeline: view();
    }
}

@media (min-width: 901px) and (max-width: 1120px) {
    .page-masthead {
        grid-template-columns: minmax(0, 1.18fr) minmax(360px, 0.82fr);
    }

    .hero-copy {
        padding: 3.35rem 1rem 3.35rem 2.8rem;
    }

    .page-masthead h1 {
        font-size: clamp(2.55rem, 4.3vw, 2.95rem);
        letter-spacing: -0.046em;
        line-height: 1.01;
    }

    .page-masthead p {
        font-size: 0.96rem;
    }

    .hero-scene {
        padding: 1.2rem;
    }

    .scene-frame {
        max-width: 360px;
    }
}

@media (max-width: 900px) {
    .block-container,
    [data-testid="stMainBlockContainer"] {
        padding: 1rem 1rem 3rem !important;
    }

    .page-masthead {
        grid-template-columns: 1fr;
    }

    .hero-copy {
        padding: 3rem 2rem 1.4rem;
    }

    .hero-scene {
        min-height: 390px;
        padding: 1.5rem 2rem 3rem;
    }

    .scene-frame {
        max-width: 340px;
    }

    .system-story {
        gap: 1.5rem;
        grid-template-columns: 1fr;
    }

    .system-story::before {
        display: none;
    }

    .story-map {
        min-height: 500px;
        position: relative;
        top: auto;
    }

    .story-map-graphic {
        height: 350px;
        left: 50%;
        max-width: 360px;
        transform: translateX(-50%) translateZ(18px);
        width: calc(100% - 3rem);
    }

    div[data-testid="stHorizontalBlock"]:has(.feature-panel) {
        flex-direction: column;
        gap: 1rem;
    }

    div[data-testid="stHorizontalBlock"]:has(.feature-panel) > div[data-testid="stColumn"] {
        flex: 1 1 100%;
        width: 100%;
    }

    .feature-panel {
        min-height: 0;
    }

    .story-steps {
        gap: 1rem;
        padding: 0 0 0.5rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        margin-left: 0.75rem;
        margin-right: 0.75rem;
    }

    .stat-band,
    .ledger-grid {
        grid-template-columns: 1fr;
    }

    .stat-item + .stat-item,
    .ledger-cell + .ledger-cell {
        border-left: 0;
        border-top: 1px solid var(--line);
    }

    .stat-item {
        min-height: auto;
    }

    .risk-topline {
        align-items: flex-start;
        flex-direction: column;
    }
}

@media (min-width: 681px) and (max-width: 900px) {
    .page-masthead {
        grid-template-columns: minmax(0, 1.18fr) minmax(245px, 0.82fr);
        min-height: 430px;
    }

    .hero-copy {
        padding: 2.75rem 1rem 2.75rem 2.25rem;
    }

    .page-masthead h1 {
        font-size: clamp(2.25rem, 5.2vw, 2.6rem);
        letter-spacing: -0.045em;
        line-height: 1.02;
    }

    .page-masthead p {
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .hero-facts {
        margin-top: 1.4rem;
        padding-top: 0.9rem;
    }

    .hero-fact {
        padding-right: 0.55rem;
    }

    .hero-fact + .hero-fact {
        padding-left: 0.55rem;
    }

    .hero-fact strong {
        font-size: 0.82rem;
    }

    .hero-fact span {
        font-size: 0.62rem;
    }

    .hero-scene {
        min-height: 360px;
        padding: 0.6rem;
    }

    .scene-frame {
        max-width: 315px;
        transform: rotateX(2deg) rotateY(-2deg) rotateZ(-2deg);
    }

    .scene-core {
        height: 110px;
        width: 110px;
    }

    .scene-core small {
        font-size: 0.5rem;
    }

    .scene-core strong {
        font-size: 0.84rem;
    }

    .scene-node {
        font-size: 0.54rem;
        letter-spacing: 0.05em;
        padding: 0.5rem 0.55rem;
    }

    .scene-caption {
        bottom: 0.7rem;
        font-size: 0.49rem;
        left: 0.75rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.1rem;
        margin-left: 0.8rem;
        margin-right: 0.8rem;
        padding: 0.32rem;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 0.72rem;
        padding: 0.64rem 0.62rem;
    }
}

@media (min-width: 681px) and (max-width: 740px) {
    .scene-core {
        height: 90px;
        width: 90px;
    }

    .scene-core small {
        font-size: 0.43rem;
    }

    .scene-core strong {
        font-size: 0.72rem;
    }

    .scene-node {
        font-size: 0.46rem;
        gap: 0.35rem;
        letter-spacing: 0.03em;
        padding: 0.42rem 0.44rem;
    }

    .scene-node::before {
        height: 2px;
        width: 8px;
    }

    .scene-caption {
        bottom: 0.6rem;
        font-size: 0.43rem;
        left: 0.65rem;
    }
}

@media (max-width: 600px) {
    .page-masthead {
        border-radius: 18px;
    }

    .hero-copy {
        padding: 2.4rem 1.35rem 1rem;
    }

    .hero-facts {
        grid-template-columns: 1fr;
    }

    .hero-fact {
        padding: 0.65rem 0;
    }

    .hero-fact + .hero-fact {
        border-left: 0;
        border-top: 1px solid rgba(207, 231, 241, 0.13);
        padding-left: 0;
    }

    .hero-scene {
        min-height: 315px;
        padding: 1rem 1.1rem 2.4rem;
    }

    .scene-frame {
        max-width: min(270px, calc(100% - 0.75rem));
        transform: none;
    }

    .scene-core {
        height: 96px;
        width: 96px;
    }

    .scene-node {
        font-size: 0.5rem;
        letter-spacing: 0.04em;
        padding: 0.48rem 0.5rem;
    }

    .scene-caption {
        bottom: 0.65rem;
        font-size: 0.48rem;
        left: 0.7rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        margin-left: 0.5rem;
        margin-right: 0.5rem;
        margin-top: -1rem;
    }

    .stTabs [data-testid="stTabsScrollLeft"],
    .stTabs [data-testid="stTabsScrollRight"] {
        display: none !important;
    }

    .story-map {
        min-height: 590px;
    }

    .story-map-graphic {
        bottom: 0.65rem;
        height: 330px;
    }

    .story-step {
        padding-left: 3.9rem;
    }
}

@media (hover: none), (pointer: coarse) {
    .hero-scene {
        perspective: none;
    }

    .hero-scene::before,
    .hero-scene::after,
    .scene-frame,
    .scene-frame::before,
    .scene-frame::after,
    .scene-flow,
    .scene-node,
    .scene-caption {
        transform: none;
    }

    .scene-frame {
        transform-style: flat;
    }

    .scene-core {
        transform: translate(-50%, -50%);
    }

    .scene-sweep {
        animation-name: sceneOrbitFlat;
        transform: none;
    }
}

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.001ms !important;
    }

    .hero-copy > *,
    .hero-line > span,
    .hero-fact,
    .stat-item,
    [data-testid="metric-container"],
    .stTabs [data-baseweb="tab-panel"],
    .story-step,
    .risk-panel,
    .policy-ledger,
    .scene-core,
    .scene-flow-signal,
    .scene-sweep {
        animation: none !important;
        opacity: 1 !important;
    }
}

/* Interactive Lab overrides: compact, evidence-led, and operational. */
[data-testid="stAppViewContainer"] {
    background-color: #E7EFF0;
    background-image: linear-gradient(
        to bottom,
        rgba(36, 91, 120, 0.025) 1px,
        transparent 1px
    );
    background-size: 100% 32px;
}

.lab-masthead {
    grid-template-columns: minmax(0, 1.25fr) minmax(270px, 0.75fr);
    min-height: 268px;
}

.lab-masthead .hero-copy {
    padding: 1.45rem 1.4rem 1.45rem 2rem;
}

.lab-masthead .hero-copy > * {
    animation: none;
}

.lab-masthead .hero-brand {
    min-height: 32px;
}

.lab-masthead .hero-brand-mark {
    height: 32px;
    width: 32px;
}

.lab-brand-lockup {
    display: flex;
    flex-direction: column;
    gap: 0.13rem;
}

.lab-product-mode {
    color: #8EC7DD;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.5625rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    line-height: 1.2;
    text-transform: uppercase;
}

.lab-masthead h1 {
    font-size: clamp(1.45rem, 2.6vw, 2.05rem);
    letter-spacing: -0.035em;
    line-height: 1.08;
    margin: 0.75rem 0 0.48rem;
    max-width: 650px;
}

.lab-masthead p {
    font-size: 0.9rem;
    line-height: 1.52;
    max-width: 660px;
}

.lab-masthead .hero-scene {
    min-height: 268px;
    padding: 0;
    perspective: 900px;
    transform: none;
}

.lab-masthead .scene-frame,
.lab-masthead .scene-frame:hover {
    max-width: 370px;
    transform: scale(0.72) rotateX(3deg) rotateY(-3deg) rotateZ(-3deg);
    transition-duration: 220ms;
}

.lab-masthead .scene-flow-signal,
.lab-masthead .scene-sweep,
.lab-masthead .scene-core {
    animation: none;
}

.lab-evidence-strip {
    background: rgba(247, 250, 250, 0.94);
    border: 1px solid var(--line);
    border-radius: 12px;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin: 0.8rem 0 0;
    overflow: hidden;
}

.lab-evidence-item {
    min-width: 0;
    padding: 0.85rem 1rem;
}

.lab-evidence-item + .lab-evidence-item {
    border-left: 1px solid var(--line);
}

.lab-evidence-value {
    color: var(--ink);
    display: block;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 1rem;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    line-height: 1.25;
}

.lab-evidence-label {
    color: var(--muted);
    display: block;
    font-size: 0.68rem;
    line-height: 1.35;
    margin-top: 0.18rem;
}

.lab-synthetic-notice {
    background: rgba(247, 250, 250, 0.78);
    border-left: 3px solid var(--atlantic);
    color: var(--muted);
    font-size: 0.78rem;
    line-height: 1.5;
    margin: 0.65rem 0 0.1rem;
    padding: 0.62rem 0.85rem;
}

.lab-synthetic-notice strong {
    color: var(--ink);
    font-weight: 600;
}

.stTabs [data-baseweb="tab-list"] {
    backdrop-filter: none;
    background: rgba(247, 250, 250, 0.96);
    border-color: var(--line);
    border-radius: 10px;
    box-shadow: none;
    gap: 0.2rem;
    margin: 0.9rem 0 0;
    padding: 0.28rem;
}

.stTabs [data-baseweb="tab"] {
    flex: 1 1 0;
    justify-content: center;
    min-width: max-content;
    padding: 0.72rem 0.8rem;
    transition: background-color 160ms cubic-bezier(0, 0, 0.38, 0.9),
                color 160ms cubic-bezier(0, 0, 0.38, 0.9),
                transform 160ms cubic-bezier(0, 0, 0.38, 0.9);
}

.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"]:hover {
    background: var(--deep-ocean) !important;
    color: #FFFFFF !important;
    transform: none;
}

.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-panel"] {
    animation: none;
    padding-top: 1.05rem;
}

.story-progress,
.story-node,
.story-step {
    animation: none !important;
    opacity: 1 !important;
}

@media (max-width: 900px) {
    .lab-masthead {
        grid-template-columns: minmax(0, 1.25fr) minmax(210px, 0.75fr);
        min-height: 240px;
    }

    .lab-masthead .hero-copy {
        padding: 1.35rem 0.5rem 1.35rem 1.5rem;
    }

    .lab-masthead .hero-scene {
        min-height: 240px;
        padding: 0;
    }

    .lab-masthead .scene-frame,
    .lab-masthead .scene-frame:hover {
        max-width: 330px;
        transform: scale(0.68) rotateX(2deg) rotateY(-2deg) rotateZ(-2deg);
    }

    .stTabs [data-baseweb="tab-list"] {
        margin: 0.8rem 0 0;
    }
}

@media (max-width: 680px) {
    .lab-masthead {
        display: block;
        min-height: 0;
    }

    .lab-masthead .hero-copy {
        padding: 1.35rem 1.25rem;
    }

    .lab-masthead .hero-scene {
        display: none;
    }

    .lab-masthead h1 {
        font-size: 1.42rem;
    }

    .lab-evidence-strip {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .lab-evidence-item:nth-child(3) {
        border-left: 0;
    }

    .lab-evidence-item:nth-child(n + 3) {
        border-top: 1px solid var(--line);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.2rem;
        margin: 0.8rem 0 0;
        overflow-x: auto;
        padding: 0.25rem;
        scroll-padding-inline: 0.25rem;
    }

    .stTabs [data-baseweb="tab"] {
        flex: 0 0 auto;
        font-size: 0.76rem;
        padding: 0.66rem 0.72rem;
    }
}

@media (max-width: 480px) {
    .lab-masthead .hero-brand-mark {
        height: 28px;
        width: 28px;
    }

    .lab-product-mode {
        display: none;
    }
}
</style>
""", unsafe_allow_html=True)

# Load the versioned premium product layer after the legacy shell styles. A CSS
# Path is rendered as a style element by Streamlit 1.58 without adding layout space.
st.html(PREMIUM_CSS_PATH)

MODEL_PATH = os.path.join('models', 'xgboost_churn_model.pkl')
DATA_PATH = os.path.join('data', 'irish_banking_churn.csv')

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_PATH):
    st.error("Model file or dataset not found. Run models/train_model.py first.")
    st.stop()

@st.cache_resource
def get_phase1_runtime(model_path):
    return load_phase1_runtime(model_path)


phase1_runtime = get_phase1_runtime(MODEL_PATH)
xgb_model = phase1_runtime.model
encoders = phase1_runtime.encoders
feature_names = list(phase1_runtime.feature_names)
continuous_features = list(phase1_runtime.continuous_features)

df_data = pd.read_csv(DATA_PATH)


def _render_stat_band(items):
    """Render a compact row of comparable facts."""
    cells = []
    for item in items:
        tone = item.get("tone", "")
        cells.append(
            f'<div class="stat-item {escape(tone)}">'
            f'<div class="stat-label">{escape(str(item["label"]))}</div>'
            f'<div class="stat-value">{escape(str(item["value"]))}</div>'
            f'<div class="stat-note">{escape(str(item["note"]))}</div>'
            "</div>"
        )
    st.markdown(
        f'<div class="stat-band" style="--stat-count:{len(items)}">'
        + "".join(cells)
        + "</div>",
        unsafe_allow_html=True,
    )


def _style_plot(fig, height=340):
    """Apply the shared Atlantic Ledger chart treatment."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans, Arial, sans-serif", color="#071827", size=12),
        title=dict(font=dict(size=15, color="#071827"), x=0.02, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#52636E"),
            title_font=dict(color="#52636E"),
        ),
        margin=dict(l=58, r=24, t=54, b=56),
        hoverlabel=dict(
            bgcolor="#071827",
            bordercolor="#071827",
            font=dict(color="#F4F1E8", family="IBM Plex Sans, Arial, sans-serif"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#D7E1E6",
        linecolor="#BDC7C9",
        tickfont=dict(color="#5C6F7E"),
        title_font=dict(color="#5C6F7E"),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#D7E1E6",
        linecolor="#BDC7C9",
        tickfont=dict(color="#5C6F7E"),
        title_font=dict(color="#5C6F7E"),
        zeroline=False,
    )
    return fig


def _display_identifier(value):
    return str(value).replace("_", " ").strip().title()


def _natural_prose(value):
    return (
        str(value)
        .replace("—", ",")
        .replace("–", " to ")
    )


def _render_policy_ledger(recommendation):
    verdict = recommendation.get("checker_verdict", "blocked")
    flags = recommendation.get("regulatory_flags", [])
    needs_review = verdict == "approved" and any(
        "human_review_required" in str(flag).lower() for flag in flags
    )
    if verdict != "approved":
        ledger_class = "blocked"
        state = "Blocked by local gate"
    elif needs_review:
        ledger_class = "review"
        state = "Passed, advisor review required"
    else:
        ledger_class = ""
        state = "Passed local checks"

    flag_text = ", ".join(str(flag) for flag in flags) if flags else "No flags"
    action = _display_identifier(recommendation.get("action", "no recommendation"))
    justification = _natural_prose(
        recommendation.get("justification", "No justification supplied.")
    )
    confidence = float(recommendation.get("confidence", 0))
    st.markdown(
        f'<div class="policy-ledger {ledger_class}">'
        '<div class="ledger-verdict">'
        '<div class="ledger-title">Local policy decision</div>'
        f'<div class="ledger-state">{escape(state)}</div>'
        "</div>"
        '<div class="ledger-grid">'
        '<div class="ledger-cell">'
        '<div class="ledger-label">Proposed action</div>'
        f'<div class="ledger-value">{escape(action)}</div>'
        "</div>"
        '<div class="ledger-cell">'
        '<div class="ledger-label">Decision record</div>'
        f'<div class="ledger-copy">{escape(justification)}</div>'
        "</div>"
        '<div class="ledger-cell">'
        '<div class="ledger-label">Agent confidence, not calibrated</div>'
        f'<div class="ledger-value">{confidence:.0%}</div>'
        "</div>"
        "</div>"
        f'<div class="ledger-footer">Local policy record: {escape(flag_text)}</div>'
        "</div>",
        unsafe_allow_html=True,
    )


def _trace_call_summary(name: str, inp: dict) -> str:
    """One line label for a tool call step."""
    if name == "product_lookup":
        category = _display_identifier(inp.get("category", "all"))
        return f"Search available products in {category}"
    if name == "segment_comparison":
        return "Compare this customer with a similar cohort"
    if name == "regulatory_constraint_checker":
        action = _display_identifier(inp.get("action_id", "proposed action"))
        review = (
            "advisor review marked as required"
            if inp.get("requires_human_review")
            else "advisor review not marked as required"
        )
        return f"Check local rules for {action}, {review}"
    if name == "recommendation_formatter":
        action = _display_identifier(inp.get("action", "proposed action"))
        confidence = inp.get("confidence")
        conf_str = (
            f", {confidence:.0%} agent confidence"
            if confidence is not None
            else ""
        )
        return f"Format {action}{conf_str}"
    return _display_identifier(name)


def _trace_result_summary(name: str, result: dict) -> str:
    """One line label for a tool result step."""
    if name == "product_lookup":
        offers = result.get("offers", [])
        if not offers:
            return "Product search found no matching offers"
        label = ", ".join(
            o.get("name", _display_identifier(o.get("action_id", "?")))
            for o in offers[:2]
        )
        suffix = f" and {len(offers) - 2} more" if len(offers) > 2 else ""
        return f"Product search found {len(offers)} offers, {label}{suffix}"
    if name == "segment_comparison":
        size = result.get("cohort_size", "?")
        rate = result.get("churn_rate")
        pred = result.get("target_phase1_prediction", {})
        risk = pred.get("churn_probability")
        rate_str = f"{rate:.1%}" if rate is not None else "?"
        risk_str = f"{risk:.1%}" if risk is not None else "?"
        return f"Cohort of {size}, churn rate {rate_str}, customer risk {risk_str}"
    if name == "regulatory_constraint_checker":
        verdict = result.get("checker_verdict", "?")
        failed = result.get("failed_rule_ids", [])
        rules = result.get("rule_results", [])
        passed_n = sum(1 for r in rules if r.get("passed"))
        if verdict == "approved":
            return f"Policy check passed all {len(rules)} rules"
        fail_str = ", ".join(failed)
        return f"Policy check blocked, {passed_n} of {len(rules)} rules passed, failed {fail_str}"
    if name == "recommendation_formatter":
        action = _display_identifier(result.get("action", "proposed action"))
        verdict = result.get("checker_verdict", "?")
        return f"Structured output for {action}, verdict {verdict}"
    return _display_identifier(name)


def _decision_state_meta(recommendation):
    state = recommendation_state(recommendation)
    return {
        "agent_ready": ("ready", "Awaiting governed run"),
        "approved": ("approved", "Passed local gate"),
        "blocked": ("blocked", "Blocked by local rule"),
        "review_required": ("review", "Advisor review required"),
    }[state]


def _render_gate_runtime_strip(
    *,
    source: str,
    api_key_available: bool,
    live_runs_remaining: int,
    session_run_cap: int,
    quota_snapshot: dict,
    model_name: str,
):
    if source == "Recorded replay":
        runtime_class = "recorded"
        runtime_label = "Recorded · zero requests"
    elif (
        api_key_available
        and live_runs_remaining > 0
        and quota_snapshot["daily_requests_remaining"] > 0
    ):
        runtime_class = "available"
        runtime_label = "Live runtime ready"
    else:
        runtime_class = "unavailable"
        runtime_label = "Live runtime unavailable"

    st.markdown(
        '<section class="gate-runtime-strip" aria-label="Agent runtime status">'
        '<div class="gate-runtime-cell">'
        '<span class="gate-runtime-label">Execution</span>'
        f'<span class="gate-runtime-value state {escape(runtime_class)}">'
        f'<span class="gate-runtime-dot"></span>{escape(runtime_label)}</span>'
        "</div>"
        '<div class="gate-runtime-cell">'
        '<span class="gate-runtime-label">Model</span>'
        f'<span class="gate-runtime-value">{escape(model_name)}</span>'
        "</div>"
        '<div class="gate-runtime-cell">'
        '<span class="gate-runtime-label">Session allowance</span>'
        f'<span class="gate-runtime-value">{live_runs_remaining}/{session_run_cap} runs</span>'
        "</div>"
        '<div class="gate-runtime-cell">'
        '<span class="gate-runtime-label">Local daily safety budget</span>'
        '<span class="gate-runtime-value">'
        f'{int(quota_snapshot["daily_requests_remaining"]):,} requests</span>'
        "</div>"
        "</section>",
        unsafe_allow_html=True,
    )


def _render_gate_case_strip(customer, *, source_label: str, recommendation):
    state_class, state_label = _decision_state_meta(recommendation)
    probability = float(customer.get("churn_probability", 0))
    st.markdown(
        '<section class="gate-case-strip" aria-label="Selected governed case">'
        '<div class="gate-case-cell">'
        '<span class="gate-case-label">Customer</span>'
        f'<span class="gate-case-value mono">{escape(str(customer.get("customer_id", "—")))}</span>'
        "</div>"
        '<div class="gate-case-cell">'
        '<span class="gate-case-label">Churn score</span>'
        f'<span class="gate-case-value mono">{probability:.1%}</span>'
        "</div>"
        '<div class="gate-case-cell">'
        '<span class="gate-case-label">Evidence source</span>'
        f'<span class="gate-case-value">{escape(source_label)}</span>'
        "</div>"
        '<div class="gate-case-cell">'
        '<span class="gate-case-label">Data boundary</span>'
        '<span class="gate-case-value">Synthetic only</span>'
        "</div>"
        '<div class="gate-case-cell">'
        '<span class="gate-case-label">Gate state</span>'
        f'<span class="gate-case-value gate-state {escape(state_class)}">'
        f'{escape(state_label)}</span>'
        "</div>"
        "</section>",
        unsafe_allow_html=True,
    )


def _gate_value(value):
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if value is None:
        return "—"
    return str(value)


def _render_gate_evidence(customer):
    profile = customer.get("profile", {})
    governance = customer.get("governance", {})
    held_products = customer.get("held_products", [])

    relationship = (
        f'{_gate_value(profile.get("account_type"))} · '
        f'{_gate_value(profile.get("tenure_months"))} '
        f'{"month" if profile.get("tenure_months") == 1 else "months"}'
    )
    transaction_count = profile.get("monthly_transaction_count")
    activity = (
        f'{_gate_value(transaction_count)} '
        f'{"transaction" if transaction_count == 1 else "transactions"} · '
        f'€{float(profile.get("monthly_transaction_amount_eur", 0)):,.0f} monthly'
    )
    products = (
        ", ".join(_display_identifier(item) for item in held_products)
        if held_products
        else "None recorded"
    )
    if profile.get("was_kbc_ulster_customer"):
        switch_difficulty = (
            "difficulty recorded"
            if profile.get("experienced_switching_difficulty")
            else "no difficulty recorded"
        )
        months_since_switching = profile.get("months_since_switching")
        switching = (
            f'Former KBC/Ulster · {_gate_value(months_since_switching)} '
            f'{"month" if months_since_switching == 1 else "months"} · {switch_difficulty}'
        )
    else:
        switching = "No KBC/Ulster migration history"
    service_calls = profile.get("customer_service_calls_6months")
    complaint_copy = (
        "complaint recorded"
        if profile.get("has_complaint_history")
        else "no complaint recorded"
    )
    service = (
        f'{_gate_value(service_calls)} {"call" if service_calls == 1 else "calls"} / 6 months · '
        f'{complaint_copy}'
    )
    governance_summary = (
        f'Arrears {_gate_value(governance.get("in_arrears", False)).lower()} · '
        f'vulnerability {_gate_value(governance.get("vulnerable_customer", False)).lower()}'
    )
    dossier_rows = (
        ("Relationship", relationship),
        ("Monthly activity", activity),
        ("Held products", products),
        ("Switching history", switching),
        ("Service signals", service),
        ("Governance overlay", governance_summary),
    )
    dossier_html = "".join(
        '<div class="gate-dossier-row">'
        f'<span class="gate-dossier-label">{escape(label)}</span>'
        f'<span class="gate-dossier-value">{escape(value)}</span>'
        "</div>"
        for label, value in dossier_rows
    )

    drivers = list(customer.get("churn_drivers", []))[:5]
    max_magnitude = max(
        (abs(float(driver.get("shap_value", 0))) for driver in drivers),
        default=1.0,
    ) or 1.0
    driver_html = []
    for driver in drivers:
        shap_value = float(driver.get("shap_value", 0))
        width = max(4.0, abs(shap_value) / max_magnitude * 100)
        increases = shap_value >= 0
        direction_class = "" if increases else "decrease"
        direction_label = "Raises model score" if increases else "Lowers model score"
        name = _display_identifier(driver.get("feature", "Unknown feature"))
        customer_value = _gate_value(driver.get("value"))
        driver_html.append(
            '<div class="gate-driver-row">'
            '<div class="gate-driver-head">'
            f'<span class="gate-driver-name">{escape(name)}</span>'
            f'<span class="gate-driver-value">{escape(customer_value)} · {shap_value:+.3f}</span>'
            "</div>"
            '<div class="gate-driver-track" aria-hidden="true">'
            f'<div class="gate-driver-fill {direction_class}" style="width:{width:.1f}%"></div>'
            "</div>"
            f'<span class="gate-driver-direction">{escape(direction_label)}</span>'
            "</div>"
        )
    if not driver_html:
        driver_html.append(
            '<p class="gate-boundary-note">No local driver evidence is attached to this case.</p>'
        )

    governance_note = _natural_prose(
        customer.get(
            "governance_note",
            "All values in this workspace are synthetic demonstration data.",
        )
    )
    st.markdown(
        '<div class="gate-evidence-grid">'
        '<section class="gate-evidence-section" aria-labelledby="gate-dossier-title">'
        '<h3 class="gate-evidence-title" id="gate-dossier-title">Customer dossier</h3>'
        f"{dossier_html}"
        "</section>"
        '<section class="gate-evidence-section" aria-labelledby="gate-drivers-title">'
        '<h3 class="gate-evidence-title" id="gate-drivers-title">Leading model drivers</h3>'
        f'{"".join(driver_html)}'
        f'<p class="gate-boundary-note">{escape(governance_note)}</p>'
        "</section>"
        "</div>",
        unsafe_allow_html=True,
    )


def _gate_event_summary(event):
    event_type = event.get("type", "event")
    content = event.get("content", {})
    if event_type == "model_thought":
        return "Evidence review completed before the next tool decision"
    if event_type == "tool_call":
        return _trace_call_summary(content.get("name", "unknown"), content.get("input", {}))
    if event_type == "tool_result":
        if content.get("is_error"):
            return f'{_display_identifier(content.get("name", "tool"))} stopped safely'
        return _trace_result_summary(
            content.get("name", "unknown"), content.get("result", {})
        )
    if event_type == "gate_check":
        if content.get("passed"):
            return "Local policy gate passed the proposed action"
        failed = ", ".join(content.get("failed_rule_ids", []))
        suffix = f" · {failed}" if failed else ""
        return f"Local policy gate stopped the proposed action{suffix}"
    if event_type == "final_output":
        if content.get("checker_verdict") == "approved":
            return f'Governed output prepared for {_display_identifier(content.get("action", "action"))}'
        return "Governed refusal returned; no recommendation was issued"
    return _display_identifier(event_type)


def _render_gate_stage_detail(stage, *, trace_is_recorded: bool):
    status_labels = {
        "complete": "Complete",
        "blocked": "Governed block",
        "review_required": "Advisor review",
        "pending": "Pending",
        "unavailable": "Unavailable",
    }
    event_rows = "".join(
        '<div class="gate-event-row">'
        f'<span class="gate-event-kind">Step {int(event.get("step", 0)):02d}</span>'
        f'<span class="gate-event-copy">{escape(_gate_event_summary(event))}</span>'
        "</div>"
        for event in stage["events"]
    )
    source_copy = (
        "Recorded fixture · zero provider requests"
        if trace_is_recorded
        else "Live Groq run · local tools and policy gate"
    )
    st.markdown(
        '<section class="gate-stage-panel" aria-label="Selected decision stage">'
        '<div class="gate-stage-heading">'
        "<div>"
        f'<span class="gate-stage-kicker">{escape(source_copy)}</span>'
        f'<div class="gate-stage-title">{escape(stage["summary"])}</div>'
        "</div>"
        f'<span class="gate-stage-status {escape(stage["status"])}">'
        f'{escape(status_labels[stage["status"]])}</span>'
        "</div>"
        f'<div class="gate-event-list">{event_rows}</div>'
        "</section>",
        unsafe_allow_html=True,
    )

    gate_events = [
        event for event in stage["events"] if event.get("type") == "gate_check"
    ]
    if gate_events:
        rule_rows = []
        for result in gate_events[-1].get("content", {}).get("rule_results", []):
            passed = bool(result.get("passed"))
            status_class = "" if passed else "blocked"
            status_word = "Passed" if passed else "Stopped"
            rule_rows.append(
                '<div class="gate-rule-row">'
                f'<span class="gate-rule-id {status_class}">'
                f'{escape(str(result.get("rule_id", "—")))}</span>'
                f'<span class="gate-rule-copy"><strong>{status_word}.</strong> '
                f'{escape(_natural_prose(result.get("reason", "No reason supplied.")))}</span>'
                "</div>"
            )
        st.markdown("".join(rule_rows), unsafe_allow_html=True)

    with st.expander("Inspect raw stage events"):
        st.caption(
            "Complete trace payload for audit inspection. Recorded analysis text is a "
            "scripted fixture; live analysis text is not treated as policy evidence."
        )
        st.json(stage["events"])


class XGBoostClassifierWrapper:
    """
    Casts DataFrame columns back to their training dtypes before each prediction.
    DiCE mutates column types during counterfactual search, which breaks the native
    XGBoost predictor without this guard.
    """
    def __init__(self, model, feature_names, dtypes):
        self.model = model
        self.feature_names = feature_names
        self.dtypes = dtypes
        self.classes_ = model.classes_

    def predict_proba(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_cast = X.copy()
        for col in self.feature_names:
            X_cast[col] = pd.to_numeric(X_cast[col], errors='coerce').fillna(0).astype(self.dtypes[col])
        return self.model.predict_proba(X_cast)

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        X_cast = X.copy()
        for col in self.feature_names:
            X_cast[col] = pd.to_numeric(X_cast[col], errors='coerce').fillna(0).astype(self.dtypes[col])
        return self.model.predict(X_cast)


@st.cache_data
def get_test_predictions():
    df_model = df_data.drop(columns=['customer_id'])
    categorical_cols = ['account_type', 'credit_score_band']
    boolean_cols = [
        'has_direct_debits', 'uses_digital_bank_secondary', 'was_kbc_ulster_customer',
        'experienced_switching_difficulty', 'has_complaint_history', 'has_mortgage', 'has_savings_goal'
    ]
    for col in categorical_cols:
        df_model[col] = encoders[col].transform(df_model[col])
    for col in boolean_cols:
        df_model[col] = df_model[col].astype(int)

    X = df_model.drop(columns=['churn'])
    y = df_model['churn']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    y_prob = xgb_model.predict_proba(X_test)[:, 1]
    return X_test, y_test, y_prob, X.dtypes


X_test, y_test, y_prob, train_dtypes = get_test_predictions()
y_pred = xgb_model.predict(X_test)
xgb_f1 = float(f1_score(y_test, y_pred))
xgb_roc_auc = float(roc_auc_score(y_test, y_prob))
xgb_average_precision = float(average_precision_score(y_test, y_prob))
xgb_confusion_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])


@st.cache_resource
def init_dice_explainer():
    template_df = X_test.copy()
    template_df['churn'] = y_test
    d = dice_ml.Data(
        dataframe=template_df,
        continuous_features=continuous_features,
        outcome_name='churn'
    )
    wrapped_model = XGBoostClassifierWrapper(xgb_model, feature_names, train_dtypes)
    m = dice_ml.Model(model=wrapped_model, backend="sklearn")
    return dice_ml.Dice(d, m, method="random")


dice_explainer = init_dice_explainer()


@st.cache_resource
def get_shap_explainer():
    """Reuse the fitted model explainer across assessments and reruns."""
    return shap.TreeExplainer(xgb_model)


CASE_WIDGET_DEFAULTS = {
    "case_customer_reference": "ATL-DEMO-001",
    "case_age": 42,
    "case_tenure_months": 24,
    "case_account_type": "Current Account",
    "case_monthly_balance_eur": 2500,
    "case_num_products": 2,
    "case_monthly_transaction_count": 45,
    "case_monthly_transaction_amount_eur": 1200,
    "case_has_direct_debits": "Yes",
    "case_direct_debit_count": 4,
    "case_uses_digital_bank_secondary": "No",
    "case_was_kbc_ulster_customer": "No",
    "case_months_since_switching": 12,
    "case_experienced_switching_difficulty": "No",
    "case_branch_visits_monthly": 1,
    "case_customer_service_calls_6months": 1,
    "case_has_complaint_history": "No",
    "case_credit_score_band": "Medium",
    "case_has_mortgage": "No",
    "case_has_savings_goal": "No",
}


def _initialize_case_widgets():
    for key, value in CASE_WIDGET_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def _sync_mortgage_from_account_type():
    st.session_state["case_has_mortgage"] = (
        "Yes"
        if st.session_state.get("case_account_type") == "Current + Mortgage"
        else "No"
    )


def _sync_account_type_from_mortgage():
    has_mortgage = st.session_state.get("case_has_mortgage") == "Yes"
    if has_mortgage:
        st.session_state["case_account_type"] = "Current + Mortgage"
    elif st.session_state.get("case_account_type") == "Current + Mortgage":
        st.session_state["case_account_type"] = "Current Account"


def _sync_direct_debits():
    if st.session_state.get("case_has_direct_debits") != "Yes":
        st.session_state["case_direct_debit_count"] = 0
    elif not st.session_state.get("case_direct_debit_count"):
        st.session_state["case_direct_debit_count"] = 4


def _sync_switching_history():
    if st.session_state.get("case_was_kbc_ulster_customer") != "Yes":
        st.session_state["case_experienced_switching_difficulty"] = "No"


def _format_case_feature_value(feature, value):
    boolean_features = {
        "has_direct_debits",
        "uses_digital_bank_secondary",
        "was_kbc_ulster_customer",
        "experienced_switching_difficulty",
        "has_complaint_history",
        "has_mortgage",
        "has_savings_goal",
    }
    if feature in encoders:
        try:
            encoded_value = int(float(value))
            return str(encoders[feature].inverse_transform([encoded_value])[0])
        except (TypeError, ValueError):
            return str(value)
    if feature in boolean_features:
        if isinstance(value, str):
            try:
                value = bool(int(float(value)))
            except ValueError:
                value = value.strip().lower() in {"yes", "true"}
        return "Yes" if bool(value) else "No"
    if feature in {"monthly_balance_eur", "monthly_transaction_amount_eur"}:
        return f"€{float(value):,.0f}"
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return f"{int(value):,}"
    return str(value)


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def _generate_counterfactual_changes(input_fingerprint, encoded_profile_json):
    """Generate bounded, cached scenario diffs for a scored profile."""
    # input_fingerprint intentionally participates in the Streamlit cache key.
    if not input_fingerprint:
        return []
    encoded_profile = json.loads(encoded_profile_json)
    input_df = pd.DataFrame([encoded_profile], columns=feature_names)
    for feature in feature_names:
        input_df[feature] = (
            pd.to_numeric(input_df[feature], errors="coerce")
            .fillna(0)
            .astype(train_dtypes[feature])
        )
    cf = dice_explainer.generate_counterfactuals(
        input_df,
        total_CFs=3,
        desired_class=0,
        features_to_vary=[
            name
            for name in feature_names
            if name
            not in {
                "age",
                "was_kbc_ulster_customer",
                "experienced_switching_difficulty",
            }
        ],
        random_seed=17,
    )
    if not cf or not cf.cf_examples_list:
        return []
    cf_df = cf.cf_examples_list[0].final_cfs_df
    if cf_df is None or cf_df.empty:
        return []

    original = input_df.iloc[0]
    changes = []
    for feature in feature_names:
        original_value = original[feature]
        candidate_values = []
        for value in cf_df[feature].tolist():
            try:
                unchanged = bool(np.isclose(float(value), float(original_value)))
            except (TypeError, ValueError):
                unchanged = value == original_value
            if unchanged:
                continue
            formatted = _format_case_feature_value(feature, value)
            if formatted not in candidate_values:
                candidate_values.append(formatted)
        if candidate_values:
            changes.append(
                {
                    "feature": feature,
                    "label": feature.replace("_", " ").title(),
                    "current_value": _format_case_feature_value(
                        feature, original_value
                    ),
                    "candidate_input": " or ".join(candidate_values),
                }
            )
    return changes


def _score_case_review(customer_id, ordered_profile, input_fingerprint):
    phase1_customer = build_phase1_customer(customer_id, ordered_profile)
    phase1_prediction = predict_customer_churn_risk(
        phase1_customer,
        phase1_runtime=phase1_runtime,
    )
    phase1_customer["churn_probability"] = phase1_prediction["churn_probability"]
    phase1_customer["phase1_prediction"] = phase1_prediction

    input_df = phase1_runtime.prepare_feature_vector(phase1_customer)
    shap_val = get_shap_explainer()(input_df)
    shap_values = np.asarray(shap_val.values[0], dtype=float)
    top_driver_indices = np.argsort(np.abs(shap_values))[::-1][:5]
    drivers = [
        {
            "feature": feature_names[index],
            "value": ordered_profile[feature_names[index]],
            "shap_value": float(shap_values[index]),
            "direction": (
                "increases_churn"
                if shap_values[index] >= 0
                else "decreases_churn"
            ),
        }
        for index in top_driver_indices
    ]
    phase1_customer["churn_drivers"] = drivers
    probability = float(phase1_prediction["churn_probability"])
    band = risk_band_for_probability(probability)
    base_value = float(np.asarray(shap_val.base_values).reshape(-1)[0])

    return CaseAssessment(
        customer_id=phase1_customer["customer_id"],
        ordered_profile=dict(ordered_profile),
        probability=probability,
        band=band.key,
        drivers=drivers,
        counterfactuals=[],
        input_fingerprint=input_fingerprint,
        model_provenance={
            "model_artifact": phase1_prediction.get("model_artifact", "Phase 1 model"),
            "prediction_method": phase1_prediction.get("prediction_method", "predict_proba"),
            "input_count": len(phase1_prediction.get("feature_columns", feature_names)),
        },
        created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        customer_payload=phase1_customer,
        shap_values=shap_values.tolist(),
        shap_base_value=base_value,
        shap_input_values=np.asarray(shap_val.data[0], dtype=float).tolist(),
        state=LabState.SCORED,
    )


def _render_ready_case_state():
    render_decision_instrument(
        variant="assessment",
        stage="draft",
        score=None,
        thresholds={"lower": 0.30, "higher": 0.60},
        steps=[
            {
                "id": "profile",
                "label": "Profile",
                "summary": "Nineteen ordered synthetic inputs are ready to assemble.",
                "detail": "Complete the three assessment sections, then run one bounded model assessment.",
                "status": "active",
            },
            {
                "id": "model",
                "label": "Model",
                "summary": "The fitted model will return a churn probability.",
                "status": "pending",
            },
            {
                "id": "explanation",
                "label": "Explanation",
                "summary": "Five signed drivers and technical SHAP evidence will follow.",
                "status": "pending",
            },
            {
                "id": "policy",
                "label": "Policy",
                "summary": "Only a current result can enter the governed Decision gate.",
                "status": "pending",
            },
        ],
        selected_step="profile",
        provenance=[
            {"label": "Record", "value": "Synthetic draft"},
            {"label": "Inputs", "value": "19 ordered fields"},
            {"label": "Boundary", "value": "Research only"},
        ],
        key="case_ready_instrument",
    )
    st.caption(
        "The result will include probability bands, local SHAP evidence, model provenance, "
        "and optional candidate scenarios. No real customer data is used."
    )


def _render_case_assessment(assessment, current_fingerprint):
    freshness = assessment_state_for_draft(assessment, current_fingerprint)
    assessment = assessment.with_state(freshness)
    st.session_state["case_assessment"] = assessment.to_session()
    is_stale = freshness is LabState.STALE

    if is_stale:
        st.session_state.pop("phase1_selected_customer", None)
        st.warning(
            "Profile changed — recalculate before this case can enter the Decision gate. "
            "The previous evidence remains visible for comparison."
        )
    else:
        st.session_state["phase1_selected_customer"] = assessment.customer_payload

    band = risk_band_for_probability(assessment.probability)
    provenance = assessment.model_provenance
    case_instrument_key = f"case_scored_instrument_{assessment.input_fingerprint[:12]}"
    initial_case_step = None if case_instrument_key in st.session_state else "model"
    render_decision_instrument(
        variant="assessment",
        stage="stale" if is_stale else "scored",
        score=assessment.probability,
        thresholds={"lower": 0.30, "higher": 0.60},
        steps=[
            {
                "id": "profile",
                "label": "Profile",
                "summary": "19/19 ordered inputs normalized at the model boundary.",
                "detail": "The visible profile and every serialized feature passed the unchanged Phase 1 schema.",
                "status": "review" if is_stale else "complete",
            },
            {
                "id": "model",
                "label": "Model",
                "summary": band.label,
                "detail": band.message,
                "status": "complete",
            },
            {
                "id": "explanation",
                "label": "Explanation",
                "summary": "Five signed local drivers are available below.",
                "detail": "Technical SHAP evidence is retained in the inspection disclosure.",
                "status": "complete",
            },
            {
                "id": "policy",
                "label": "Policy",
                "summary": (
                    "Profile changed — recalculate before handoff."
                    if is_stale
                    else "Current assessment is eligible for the Decision gate."
                ),
                "status": "review" if is_stale else "complete",
            },
        ],
        selected_step=initial_case_step,
        provenance=[
            {"label": "Record", "value": assessment.customer_id},
            {
                "label": "Model",
                "value": str(provenance.get("model_artifact", "Phase 1 model")),
            },
            {
                "label": "Inputs",
                "value": f"{int(provenance.get('input_count', 19))}/19 ordered",
            },
            {"label": "Scored", "value": assessment.created_at},
        ],
        key=case_instrument_key,
    )
    st.caption(
        f"{band.message} The 30% and 60% markers are local display thresholds, not "
        "calibrated or externally validated risk categories."
    )

    st.markdown("#### Leading model signals")
    max_effect = max(
        (abs(float(driver["shap_value"])) for driver in assessment.drivers),
        default=1.0,
    )
    driver_rows = []
    for driver in assessment.drivers:
        effect = float(driver["shap_value"])
        direction_class = "raises" if effect >= 0 else "lowers"
        direction_label = "Raises score" if effect >= 0 else "Lowers score"
        width = max(8.0, abs(effect) / max_effect * 100.0)
        feature = str(driver["feature"])
        label = feature.replace("_", " ").title()
        value = _format_case_feature_value(feature, driver.get("value"))
        driver_rows.append(
            f'<div class="case-driver {direction_class}">'
            f'<div class="case-driver-copy"><strong>{escape(label)}</strong>'
            f'<span>{escape(value)} · {direction_label}</span></div>'
            f'<div class="case-driver-track" aria-hidden="true"><span style="width:{width:.1f}%"></span></div>'
            f'<code>{effect:+.3f}</code></div>'
        )
    st.markdown(
        '<div class="case-driver-list">' + "".join(driver_rows) + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Signed SHAP values describe movement in the fitted model output. They are not "
        "probability percentage points and do not establish cause."
    )

    st.markdown(
        f"""
        <div class="case-provenance" aria-label="Assessment provenance">
          <div><span>Record</span><strong>{escape(assessment.customer_id)}</strong></div>
          <div><span>Model</span><strong>{escape(str(provenance.get('model_artifact', 'Phase 1 model')))}</strong></div>
          <div><span>Inputs</span><strong>{int(provenance.get('input_count', 19))}/19</strong></div>
          <div><span>Scored</span><strong>{escape(assessment.created_at)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Inspect technical explanation"):
        if assessment.shap_values and assessment.shap_base_value is not None:
            technical_explanation = shap.Explanation(
                values=np.asarray(assessment.shap_values, dtype=float),
                base_values=float(assessment.shap_base_value),
                data=np.asarray(assessment.shap_input_values, dtype=float),
                feature_names=feature_names,
            )
            with plt.rc_context(SHAP_PLOT_STYLE):
                fig, ax = plt.subplots(figsize=(6.2, 3.8))
                fig.patch.set_facecolor("#F4F1E8")
                ax.set_facecolor("#F4F1E8")
                shap.plots.waterfall(technical_explanation, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)
            st.caption(
                "Red marks moved the output toward churn; blue marks moved it toward "
                "retention. This local explanation is inspection evidence only."
            )

    if assessment.probability > 0.50:
        st.markdown("#### Candidate model scenarios")
        st.caption(
            "These scenarios test model inputs. They do not show that changing a customer "
            "circumstance would prevent churn."
        )
        generate_disabled = is_stale
        if st.button(
            "Generate candidate scenarios",
            key="case_generate_counterfactuals",
            disabled=generate_disabled,
            width="stretch",
        ):
            encoded_profile = assessment.customer_payload["phase1_prediction"][
                "encoded_feature_vector"
            ]
            try:
                with st.spinner("Testing bounded candidate inputs…"):
                    changes = _generate_counterfactual_changes(
                        assessment.input_fingerprint,
                        json.dumps(encoded_profile, sort_keys=True),
                    )
                assessment = assessment.with_counterfactuals(changes)
                assessment.customer_payload["counterfactuals"] = changes
                st.session_state["case_assessment"] = assessment.to_session()
                st.session_state["phase1_selected_customer"] = assessment.customer_payload
            except Exception:
                st.error(
                    "Candidate scenarios could not be generated for this profile. The "
                    "assessment and Decision gate handoff remain available."
                )
        if assessment.counterfactuals:
            scenario_rows = []
            for change in assessment.counterfactuals:
                scenario_rows.append(
                    '<div class="case-scenario-row">'
                    f'<strong>{escape(str(change["label"]))}</strong>'
                    f'<span>{escape(str(change["current_value"]))}</span>'
                    '<span class="case-scenario-arrow" aria-hidden="true">→</span>'
                    f'<span>{escape(str(change["candidate_input"]))}</span></div>'
                )
            st.markdown(
                '<div class="case-scenario-list">' + "".join(scenario_rows) + "</div>",
                unsafe_allow_html=True,
            )
        elif st.session_state.get("case_generate_counterfactuals"):
            st.info("No distinct bounded candidate changes were found for this profile.")
    else:
        st.info(
            "This score is below the candidate-scenario threshold. The local explanation "
            "and Decision gate handoff remain available."
        )

    handoff_ready = can_handoff_assessment(assessment, current_fingerprint)
    if st.button(
        "Continue this case to Decision gate",
        key="case_continue_to_decision_gate",
        type="primary",
        disabled=not handoff_ready,
        width="stretch",
    ):
        ready_assessment = assessment.with_state(LabState.AGENT_READY)
        st.session_state["case_assessment"] = ready_assessment.to_session()
        st.session_state["phase1_selected_customer"] = ready_assessment.customer_payload
        st.session_state["lab_requested_workspace"] = "Decision gate"
        st.rerun()


def render_case_review_workspace():
    """Render the premium three-step assessment and persistent evidence panel."""
    _initialize_case_widgets()
    st.markdown(
        '<section class="workspace-intro">'
        '<span class="workspace-intro__eyebrow">Workspace 01 · Case review</span>'
        '<h2>Build and inspect one case</h2>'
        '<p>Assemble a synthetic customer profile, inspect the model evidence, and '
        "hand only a current assessment into the governed Decision gate.</p></section>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="case-step-index" aria-label="Assessment sections">
          <span><b>01</b> Identity &amp; relationship</span>
          <span><b>02</b> Financial behaviour</span>
          <span><b>03</b> Switching &amp; service</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_input, col_output = st.columns([5, 7], gap="large")
    with col_input:
        st.markdown(
            '<div class="case-step-heading"><span>01</span><div><strong>Identity &amp; relationship</strong><small>Who this synthetic case represents</small></div></div>',
            unsafe_allow_html=True,
        )
        customer_reference = st.text_input(
            "Customer reference",
            key="case_customer_reference",
            help="Synthetic identifier carried unchanged into the Decision gate.",
        )
        age_col, tenure_col = st.columns(2)
        with age_col:
            age = st.number_input(
                "Age", min_value=18, max_value=75, step=1, key="case_age"
            )
        with tenure_col:
            tenure_months = st.number_input(
                "Tenure (months)",
                min_value=1,
                max_value=180,
                step=1,
                key="case_tenure_months",
            )
        account_col, mortgage_col = st.columns([1.25, 0.75])
        with account_col:
            account_type = st.selectbox(
                "Account type",
                options=[
                    "Current Account",
                    "Savings Account",
                    "Current + Savings",
                    "Current + Mortgage",
                ],
                key="case_account_type",
                on_change=_sync_mortgage_from_account_type,
            )
        with mortgage_col:
            has_mortgage_choice = st.segmented_control(
                "Mortgage",
                options=["No", "Yes"],
                key="case_has_mortgage",
                on_change=_sync_account_type_from_mortgage,
            )
        num_products = st.number_input(
            "Products held",
            min_value=1,
            max_value=5,
            step=1,
            key="case_num_products",
        )

        st.markdown(
            '<div class="case-step-heading"><span>02</span><div><strong>Financial behaviour</strong><small>Balances, activity and commitments</small></div></div>',
            unsafe_allow_html=True,
        )
        balance_col, credit_col = st.columns([1.15, 0.85])
        with balance_col:
            monthly_balance_eur = st.number_input(
                "Monthly balance (EUR)",
                min_value=0,
                max_value=50000,
                step=100,
                key="case_monthly_balance_eur",
            )
        with credit_col:
            credit_score_band = st.selectbox(
                "Credit score band",
                options=["Low", "Medium", "High"],
                key="case_credit_score_band",
            )
        transactions_col, spend_col = st.columns(2)
        with transactions_col:
            monthly_transaction_count = st.number_input(
                "Monthly transactions",
                min_value=5,
                max_value=200,
                step=1,
                key="case_monthly_transaction_count",
            )
        with spend_col:
            monthly_transaction_amount_eur = st.number_input(
                "Monthly spend (EUR)",
                min_value=100,
                max_value=8000,
                step=50,
                key="case_monthly_transaction_amount_eur",
            )
        debit_choice_col, debit_count_col = st.columns([0.95, 1.05])
        with debit_choice_col:
            has_direct_debits_choice = st.segmented_control(
                "Direct debits",
                options=["No", "Yes"],
                key="case_has_direct_debits",
                on_change=_sync_direct_debits,
            )
        with debit_count_col:
            direct_debit_count = st.number_input(
                "Direct debit count",
                min_value=0,
                max_value=15,
                step=1,
                key="case_direct_debit_count",
                disabled=has_direct_debits_choice != "Yes",
            )
        has_savings_goal_choice = st.segmented_control(
            "Active savings goal",
            options=["No", "Yes"],
            key="case_has_savings_goal",
        )

        st.markdown(
            '<div class="case-step-heading"><span>03</span><div><strong>Switching &amp; service</strong><small>Migration, channels and service pressure</small></div></div>',
            unsafe_allow_html=True,
        )
        was_kbc_ulster_choice = st.segmented_control(
            "Former KBC or Ulster Bank customer",
            options=["No", "Yes"],
            key="case_was_kbc_ulster_customer",
            on_change=_sync_switching_history,
        )
        if was_kbc_ulster_choice == "Yes":
            switching_months_col, switching_difficulty_col = st.columns(2)
            with switching_months_col:
                months_since_switching = st.number_input(
                    "Months since switching",
                    min_value=1,
                    max_value=36,
                    step=1,
                    key="case_months_since_switching",
                )
            with switching_difficulty_col:
                experienced_switching_choice = st.segmented_control(
                    "Switching difficulty",
                    options=["No", "Yes"],
                    key="case_experienced_switching_difficulty",
                )
        else:
            months_since_switching = 0
            experienced_switching_choice = "No"

        secondary_col, complaint_col = st.columns(2)
        with secondary_col:
            uses_secondary_choice = st.segmented_control(
                "Secondary digital bank",
                options=["No", "Yes"],
                key="case_uses_digital_bank_secondary",
                help="Uses Revolut or N26 as a secondary provider.",
            )
        with complaint_col:
            complaint_choice = st.segmented_control(
                "Complaint on record",
                options=["No", "Yes"],
                key="case_has_complaint_history",
            )
        branch_col, service_col = st.columns(2)
        with branch_col:
            branch_visits_monthly = st.number_input(
                "Branch visits / month",
                min_value=0,
                max_value=8,
                step=1,
                key="case_branch_visits_monthly",
            )
        with service_col:
            customer_service_calls_6months = st.number_input(
                "Service calls / 6 months",
                min_value=0,
                max_value=12,
                step=1,
                key="case_customer_service_calls_6months",
            )

        raw_profile = {
            "age": int(age),
            "tenure_months": int(tenure_months),
            "account_type": account_type,
            "monthly_balance_eur": float(monthly_balance_eur),
            "num_products": int(num_products),
            "monthly_transaction_count": int(monthly_transaction_count),
            "monthly_transaction_amount_eur": float(
                monthly_transaction_amount_eur
            ),
            "has_direct_debits": has_direct_debits_choice == "Yes",
            "direct_debit_count": int(direct_debit_count),
            "uses_digital_bank_secondary": uses_secondary_choice == "Yes",
            "was_kbc_ulster_customer": was_kbc_ulster_choice == "Yes",
            "months_since_switching": int(months_since_switching),
            "experienced_switching_difficulty": (
                experienced_switching_choice == "Yes"
            ),
            "branch_visits_monthly": int(branch_visits_monthly),
            "customer_service_calls_6months": int(
                customer_service_calls_6months
            ),
            "has_complaint_history": complaint_choice == "Yes",
            "credit_score_band": credit_score_band,
            "has_mortgage": has_mortgage_choice == "Yes",
            "has_savings_goal": has_savings_goal_choice == "Yes",
        }
        ordered_profile = normalize_case_profile(raw_profile)
        normalized_customer_id = customer_reference.strip() or "ATL-DEMO-001"
        current_fingerprint = case_input_fingerprint(
            ordered_profile, normalized_customer_id
        )
        predict_btn = st.button(
            "Assess churn risk",
            key="case_predict_churn_risk",
            type="primary",
            width="stretch",
        )

    with col_output:
        st.markdown('<div class="case-output-anchor"></div>', unsafe_allow_html=True)
        if predict_btn:
            try:
                with st.spinner("Scoring the profile and assembling model evidence…"):
                    assessment = _score_case_review(
                        normalized_customer_id,
                        ordered_profile,
                        current_fingerprint,
                    )
                st.session_state["case_assessment"] = assessment.to_session()
                st.session_state["phase1_selected_customer"] = (
                    assessment.customer_payload
                )
                st.session_state.pop("retention_live_result", None)
            except Phase1SchemaError as exc:
                st.error(f"Phase 1 feature schema validation failed: {exc}")

        assessment = CaseAssessment.from_session(
            st.session_state.get("case_assessment")
        )
        if assessment is None:
            _render_ready_case_state()
        else:
            _render_case_assessment(assessment, current_fingerprint)


CHURN_COLOR = '#A33A32'
RETAIN_COLOR = '#245B78'

st.markdown(
    f"""
    <div class="page-masthead lab-masthead">
      <div class="hero-copy">
        <div class="hero-brand">
          <span class="hero-brand-mark" aria-hidden="true">{BRAND_MARK_SVG}</span>
          <span class="lab-brand-lockup">
            <span class="hero-brand-name">Atlantic Ledger</span>
            <span class="lab-product-mode">Interactive Lab</span>
          </span>
        </div>
        <h1>Governed churn decision workbench</h1>
        <p>Review a synthetic customer, inspect the model evidence, explore lower-risk alternatives, and pass a proposed response through deterministic policy rules.</p>
      </div>
      <div class="hero-scene" role="img" aria-label="A visual map from customer signals to a governed retention decision">
        <div class="scene-frame">
          <svg class="scene-flow" viewBox="0 0 410 410" aria-hidden="true">
            <ellipse class="scene-orbit outer" cx="205" cy="205" rx="151" ry="83"></ellipse>
            <ellipse class="scene-orbit inner" cx="205" cy="205" rx="118" ry="58"></ellipse>
            <path class="scene-flow-base" d="M78 92 C142 58 268 64 333 104 C305 147 258 170 205 205 C158 237 111 264 80 322 C147 357 260 358 332 326 C295 270 259 232 205 205"></path>
            <path class="scene-flow-signal" d="M78 92 C142 58 268 64 333 104 C305 147 258 170 205 205 C158 237 111 264 80 322 C147 357 260 358 332 326 C295 270 259 232 205 205"></path>
          </svg>
          <div class="scene-axis"></div>
          <div class="scene-axis vertical"></div>
          <div class="scene-sweep"></div>
          <div class="scene-core">
            <div class="scene-core-brand" aria-hidden="true">{BRAND_MARK_SVG}</div>
            <small>Policy checked result</small>
            <strong>Pass or block</strong>
          </div>
          <div class="scene-node input">Customer signals</div>
          <div class="scene-node risk">Churn risk</div>
          <div class="scene-node reason">Model evidence</div>
          <div class="scene-node policy">Policy check</div>
          <div class="scene-caption">Prediction to governed action</div>
        </div>
      </div>
    </div>
    <section class="lab-evidence-strip" aria-label="Model and governance evidence">
      <div class="lab-evidence-item">
        <span class="lab-evidence-value">{len(df_data):,}</span>
        <span class="lab-evidence-label">synthetic customer records</span>
      </div>
      <div class="lab-evidence-item">
        <span class="lab-evidence-value">{len(feature_names)}</span>
        <span class="lab-evidence-label">ordered model inputs</span>
      </div>
      <div class="lab-evidence-item">
        <span class="lab-evidence-value">{xgb_average_precision:.3f}</span>
        <span class="lab-evidence-label">held-out average precision</span>
      </div>
      <div class="lab-evidence-item">
        <span class="lab-evidence-value">4 tools</span>
        <span class="lab-evidence-label">behind the policy gate</span>
      </div>
    </section>
    <div class="lab-synthetic-notice" role="note">
      <strong>Synthetic research environment.</strong> Customer profiles, scores, governance flags, offers, and recommendations are demonstrations—not real banking records or decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

requested_workspace = st.session_state.pop("lab_requested_workspace", None)
if requested_workspace in {
    "Case review",
    "Decision gate",
    "Model evidence",
    "Data & limits",
}:
    st.session_state["lab_workspace"] = requested_workspace

case_review, decision_gate, model_evidence, data_limits = st.tabs([
    "Case review",
    "Decision gate",
    "Model evidence",
    "Data & limits",
], key="lab_workspace", on_change="rerun")

if data_limits.open:
    with data_limits:
        render_data_limits(
            data=df_data,
            feature_count=len(feature_names),
            style_plot=_style_plot,
            churn_color=CHURN_COLOR,
            retain_color=RETAIN_COLOR,
        )

if model_evidence.open:
    with model_evidence:
        render_model_evidence(
            y_test=y_test,
            y_prob=y_prob,
            confusion=xgb_confusion_matrix,
            f1=xgb_f1,
            roc_auc=xgb_roc_auc,
            average_precision=xgb_average_precision,
            holdout_size=len(X_test),
            style_plot=_style_plot,
            assets_dir=APP_DIR / "assets",
        )





if case_review.open:
    with case_review:
        render_case_review_workspace()


def _render_decision_gate_workspace():
    st.markdown(
        '<section class="workspace-intro">'
        '<span class="workspace-intro__eyebrow">Workspace 02 · Decision gate</span>'
        '<h2>Review the governed response</h2>'
        '<p>Inspect the selected synthetic case, four bounded retention-agent stages, '
        "and the deterministic policy record before any proposed response.</p></section>",
        unsafe_allow_html=True,
    )

    demo_dir = os.path.join(os.path.dirname(__file__), "demo_traces")
    demo_records = []
    if os.path.isdir(demo_dir):
        for demo_filename in sorted(os.listdir(demo_dir)):
            if demo_filename.endswith(".json"):
                try:
                    with open(
                        os.path.join(demo_dir, demo_filename),
                        "r",
                        encoding="utf-8",
                    ) as demo_file:
                        demo_records.append(json.load(demo_file))
                except (OSError, json.JSONDecodeError) as exc:
                    st.error(f"Could not load {demo_filename}: {exc}")

    if not demo_records:
        st.error("No recorded demo traces are available.")
    else:
        # Phase 2 server-side Groq key and per-session request safety.
        from agent.loop import (
            MAX_LIVE_API_CALLS,
            MAX_LOOP_TURNS,
            MAX_TOKENS,
            MODEL_NAME,
            create_live_client,
            resolve_groq_api_key,
            run_retention_agent,
        )
        from agent.rate_limits import (
            GLOBAL_REQUEST_QUOTA,
            RateLimitSafetyError,
            SESSION_RUN_CAP,
            reserve_session_run,
        )

        if "retention_live_run_count" not in st.session_state:
            st.session_state["retention_live_run_count"] = 0

        groq_api_key = resolve_groq_api_key(st.secrets)
        live_run_count = st.session_state["retention_live_run_count"]
        live_runs_remaining = max(0, SESSION_RUN_CAP - live_run_count)
        api_key_available = bool(groq_api_key)
        quota_snapshot = GLOBAL_REQUEST_QUOTA.snapshot()

        case_review_customer = st.session_state.get("phase1_selected_customer")
        has_current_case = isinstance(case_review_customer, dict)
        current_case_id = (
            str(case_review_customer.get("customer_id")) if has_current_case else None
        )
        if has_current_case and st.session_state.get("decision_gate_case_id") != current_case_id:
            st.session_state["decision_gate_source"] = "Current case"
            st.session_state["decision_gate_case_id"] = current_case_id
        elif not has_current_case:
            st.session_state["decision_gate_source"] = "Recorded replay"

        source_options = (
            ["Current case", "Recorded replay"]
            if has_current_case
            else ["Recorded replay"]
        )
        source_col, scenario_col = st.columns([0.92, 1.48], vertical_alignment="bottom")
        with source_col:
            source = st.segmented_control(
                "Evidence source",
                options=source_options,
                key="decision_gate_source",
                width="stretch",
            )
        with scenario_col:
            if source == "Recorded replay":
                selected_title = st.selectbox(
                    "Recorded governed scenario",
                    options=[record["title"] for record in demo_records],
                    format_func=_natural_prose,
                    key="decision_gate_recording",
                )
            else:
                st.markdown(
                    '<p class="gate-boundary-note"><strong>Fresh assessment.</strong> '
                    f'The exact Case review object for <code>{escape(current_case_id)}</code> '
                    "will be passed to the live agent without rebuilding its profile.</p>",
                    unsafe_allow_html=True,
                )

        trace_is_recorded = source == "Recorded replay"
        if trace_is_recorded:
            selected_demo = next(
                record for record in demo_records if record["title"] == selected_title
            )
            customer = selected_demo["customer"]
            recommendation = selected_demo["recommendation"]
            trace = selected_demo["trace"]
        else:
            customer = case_review_customer
            recommendation = None
            trace = []
            stored_live_result = st.session_state.get("retention_live_result")
            if (
                stored_live_result
                and stored_live_result.get("customer_id") == customer["customer_id"]
            ):
                customer = stored_live_result["payload"]["customer"]
                recommendation = stored_live_result["payload"]["recommendation"]
                trace = stored_live_result["payload"]["trace"]

        _render_gate_runtime_strip(
            source=source,
            api_key_available=api_key_available,
            live_runs_remaining=live_runs_remaining,
            session_run_cap=SESSION_RUN_CAP,
            quota_snapshot=quota_snapshot,
            model_name=MODEL_NAME,
        )

        with st.expander("Runtime limits and provenance"):
            st.markdown(
                f"**Bounded live run:** up to `{MAX_LIVE_API_CALLS}` provider calls, "
                f"`{MAX_TOKENS}` tokens per call, and `{MAX_LOOP_TURNS}` loop turns. "
                "The Groq key is read only from deployment secrets or the local environment; "
                "it is never accepted or displayed by this interface."
            )
            st.caption(
                "The daily value above is a process-local safety counter. It does not read "
                "Groq account usage or requests made by another running instance. Recorded "
                "replays make zero provider requests."
            )

        if source == "Current case":
            live_disabled = (
                not api_key_available
                or live_runs_remaining <= 0
                or quota_snapshot["daily_requests_remaining"] <= 0
            )
            if st.button(
                "Run governed recommendation",
                disabled=live_disabled,
                type="primary",
                key="run_governed_recommendation",
            ):
                try:
                    GLOBAL_REQUEST_QUOTA.ensure_run_available()
                    reserve_session_run(st.session_state)
                    live_client = create_live_client(api_key=groq_api_key)
                    with st.spinner("Checking evidence, catalogue, cohort, and local rules..."):
                        live_result = run_retention_agent(
                            customer,
                            client=live_client,
                            phase1_runtime=phase1_runtime,
                        )
                    st.session_state["retention_live_result"] = {
                        "customer_id": customer["customer_id"],
                        "payload": live_result,
                    }
                    customer = live_result["customer"]
                    recommendation = live_result["recommendation"]
                    trace = live_result["trace"]
                except RateLimitSafetyError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    status_code = getattr(exc, "status_code", None)
                    error_text = str(exc).lower()
                    if status_code == 401 or "invalid_api_key" in error_text:
                        st.error(
                            "Live Groq authentication failed. The deployment owner must "
                            "replace GROQ_API_KEY in Streamlit Secrets with a valid key."
                        )
                    else:
                        st.error(f"Live run stopped safely: {exc}")

            if live_disabled and not trace:
                if not api_key_available:
                    unavailability = (
                        "No valid deployment key is configured. The current case remains "
                        "unchanged; use a recorded replay to inspect the complete governed flow."
                    )
                elif live_runs_remaining <= 0:
                    unavailability = (
                        "The five-run session allowance has been used. Recorded replays remain "
                        "available without provider requests."
                    )
                else:
                    unavailability = (
                        "The process-local daily safety budget is exhausted. No request was made."
                    )
                st.markdown(
                    '<div class="gate-governed-outcome review_required">'
                    '<strong>Live run unavailable</strong>'
                    f'<span>{escape(unavailability)}</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )

        _render_gate_case_strip(
            customer,
            source_label=("Recorded fixture" if trace_is_recorded else "Current case"),
            recommendation=recommendation,
        )

        _render_gate_evidence(customer)
        with st.expander("Inspect complete synthetic profile and model provenance"):
            st.json(
                {
                    "customer_id": customer.get("customer_id"),
                    "profile": customer.get("profile", {}),
                    "held_products": customer.get("held_products", []),
                    "governance": customer.get("governance", {}),
                    "phase1_prediction": customer.get("phase1_prediction", {}),
                }
            )

        if not trace or recommendation is None:
            st.markdown(
                '<div class="gate-governed-outcome">'
                '<strong>Case evidence is ready</strong>'
                '<span>No governed recommendation exists for this exact profile yet. '
                "Run the bounded live agent, or choose a recorded replay to inspect a "
                "complete four-tool decision.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
            return

        st.subheader("Decision timeline")
        st.markdown(
            '<p class="section-note">Select one stage to inspect its readable result. '
            "The complete source events remain available for audit without making the "
            "workspace read like a raw execution log.</p>",
            unsafe_allow_html=True,
        )
        stages = build_decision_stages(trace)
        instrument_statuses = {
            "complete": "complete",
            "blocked": "blocked",
            "review_required": "review",
            "pending": "pending",
            "unavailable": "blocked",
        }
        instrument_steps = [
            {
                "id": stage["tool_name"],
                "label": stage["label"],
                "summary": stage["summary"],
                "detail": f'{len(stage["events"])} source events retained for inspection.',
                "status": instrument_statuses[stage["status"]],
            }
            for stage in stages
        ]
        gate_events = [event for event in trace if event.get("type") == "gate_check"]
        rule_ids = [
            str(result.get("rule_id"))
            for result in (
                gate_events[-1].get("content", {}).get("rule_results", [])
                if gate_events
                else []
            )
            if result.get("rule_id")
        ]
        outcome_state = recommendation_state(recommendation)
        _, outcome_label = _decision_state_meta(recommendation)
        instrument_suffix = "".join(
            character
            for character in str(customer.get("customer_id", "case"))
            if character.isalnum()
        )[:32] or "case"
        instrument_key = f"decision_gate_instrument_{instrument_suffix}"
        initial_instrument_step = (
            None
            if instrument_key in st.session_state
            else "regulatory_constraint_checker"
        )
        selected_step_id = render_decision_instrument(
            variant="governance",
            stage=outcome_state,
            score=float(customer.get("churn_probability", 0)),
            thresholds={"lower": 0.3, "higher": 0.6},
            steps=instrument_steps,
            provenance=[
                {
                    "label": "Source",
                    "value": "Recorded replay" if trace_is_recorded else "Live Groq run",
                },
                {"label": "Model", "value": MODEL_NAME},
                {"label": "Boundary", "value": "Synthetic data only"},
                {"label": "Trace", "value": f"{len(trace)} ordered events"},
            ],
            verdict={
                "status": outcome_state,
                "label": outcome_label,
                "detail": _natural_prose(
                    recommendation.get("justification", "No justification supplied.")
                ),
            },
            rule_ids=rule_ids,
            selected_step=initial_instrument_step,
            key=instrument_key,
        )
        selected_stage = next(
            stage for stage in stages if stage["tool_name"] == selected_step_id
        )
        _render_gate_stage_detail(
            selected_stage,
            trace_is_recorded=trace_is_recorded,
        )
        with st.expander("Inspect complete trace payload"):
            st.caption(
                "All tool inputs, tool outputs, timestamps, gate results, and the final "
                "structured output are retained here in their original order."
            )
            st.json(trace)

        if outcome_state == "blocked":
            st.markdown(
                '<div class="gate-governed-outcome blocked">'
                '<strong>Governed block · no action may proceed</strong>'
                '<span>The local rule set rejected the proposed action and the formatter '
                "returned a refusal. This is an expected policy outcome, not an application "
                "failure.</span>"
                "</div>",
                unsafe_allow_html=True,
            )
        elif outcome_state == "review_required":
            st.markdown(
                '<div class="gate-governed-outcome review_required">'
                '<strong>Advisor review is part of the decision</strong>'
                '<span>The local gate passed only with the human-review record retained. '
                "The output is a proposal for advisor assessment, not an instruction to "
                "contact a customer.</span>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.subheader("Policy ledger")
        _render_policy_ledger(recommendation)
        st.caption(
            "ARR-001, HOLD-002, HUM-003, and VUL-004 are local demonstration rules. "
            "They are not legal findings and do not reproduce the Central Bank Consumer "
            "Protection Code or a bank's own eligibility controls."
        )


if decision_gate.open:
    with decision_gate:
        _render_decision_gate_workspace()
