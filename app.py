import os
import json
import warnings
from html import escape
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

st.set_page_config(
    page_title="Irish Banking Churn Predictor",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --ink: #102A43;
    --atlantic: #245B78;
    --atlantic-bright: #4EA2C6;
    --deep-ocean: #071827;
    --deep-ocean-soft: #0D2335;
    --cloud: #E4ECEE;
    --paper: #F8FBFA;
    --approval: #147D64;
    --approval-soft: #DFF3EC;
    --block: #A33A32;
    --block-soft: #F8E6E3;
    --line: #C8D6DA;
    --line-soft: #DCE6E8;
    --muted: #5C6F7E;
    --shadow-soft: 0 18px 45px rgba(16, 42, 67, 0.08);
    --shadow-lift: 0 22px 55px rgba(16, 42, 67, 0.13);
}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: "IBM Plex Sans", Arial, sans-serif;
    color: var(--ink);
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 44rem 34rem at -4% 2%, rgba(65, 145, 151, 0.2), transparent 68%),
        radial-gradient(ellipse 48rem 38rem at 104% 16%, rgba(59, 116, 145, 0.18), transparent 70%),
        linear-gradient(180deg, #E8EFF0 0%, #DDE8EA 48%, #E7EEF0 100%);
    position: relative;
}

[data-testid="stAppViewContainer"]::before {
    background-image:
        radial-gradient(circle, rgba(36, 91, 120, 0.1) 0.65px, transparent 0.85px);
    background-size: 28px 28px;
    content: "";
    height: 52rem;
    inset: 0 0 auto;
    mask-image: linear-gradient(to bottom, black 0, rgba(0, 0, 0, 0.28) 30rem, transparent 52rem);
    -webkit-mask-image: linear-gradient(to bottom, black 0, rgba(0, 0, 0, 0.28) 30rem, transparent 52rem);
    opacity: 0.22;
    pointer-events: none;
    position: absolute;
    z-index: 0;
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
    letter-spacing: -0.025em;
}

h2 {
    font-size: clamp(1.55rem, 2.4vw, 2rem) !important;
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
        radial-gradient(circle at 82% 18%, rgba(78, 162, 198, 0.2), transparent 25rem),
        var(--deep-ocean);
    border: 0;
    border-radius: 24px;
    box-shadow: 0 28px 70px rgba(7, 24, 39, 0.22), inset 0 0 0 1px rgba(191, 218, 231, 0.13);
    color: #FFFFFF;
    display: grid;
    grid-template-columns: minmax(0, 1.12fr) minmax(360px, 0.88fr);
    isolation: isolate;
    min-height: 470px;
    overflow: hidden;
    position: relative;
}

.page-masthead::before {
    background-image:
        linear-gradient(rgba(207, 231, 241, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(207, 231, 241, 0.055) 1px, transparent 1px);
    background-size: 42px 42px;
    content: "";
    inset: 0;
    mask-image: linear-gradient(90deg, transparent, black 42%, black);
    pointer-events: none;
    position: absolute;
    z-index: -1;
}

.page-masthead::after {
    background: var(--atlantic-bright);
    box-shadow: 0 0 34px rgba(78, 162, 198, 0.48);
    content: "";
    height: 2px;
    left: 0;
    position: absolute;
    top: 0;
    transform-origin: left;
    width: 38%;
}

.hero-copy {
    align-self: center;
    padding: 4rem 1.4rem 4rem 4rem;
    position: relative;
    z-index: 2;
}

.hero-kicker {
    align-items: center;
    color: #BFD8E4;
    display: flex;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.68rem;
    font-weight: 600;
    gap: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.hero-kicker::before {
    background: #68D5B3;
    border-radius: 999px;
    box-shadow: 0 0 0 5px rgba(104, 213, 179, 0.1);
    content: "";
    height: 7px;
    width: 7px;
}

.page-masthead h1 {
    color: #FFFFFF !important;
    font-size: clamp(2.6rem, 5vw, 4.9rem);
    font-weight: 500 !important;
    letter-spacing: -0.052em;
    line-height: 0.96;
    margin: 1.45rem 0 1.2rem;
    max-width: 780px;
}

.page-masthead h1 span {
    display: block;
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
    margin-top: 2.1rem;
    max-width: 670px;
    padding-top: 1.1rem;
}

.hero-fact {
    min-width: 0;
    padding-right: 1rem;
}

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
    min-height: 470px;
    overflow: hidden;
    padding: 2.5rem;
    position: relative;
}

.scene-frame {
    aspect-ratio: 1;
    border: 1px solid rgba(191, 218, 231, 0.1);
    border-radius: 28px;
    max-width: 410px;
    position: relative;
    transform: rotate(-3deg);
    width: 100%;
}

.scene-frame::before,
.scene-frame::after {
    border: 1px solid rgba(191, 218, 231, 0.12);
    border-radius: 50%;
    content: "";
    inset: 11%;
    position: absolute;
}

.scene-frame::before {
    display: none;
}

.scene-frame::after {
    border-color: rgba(78, 162, 198, 0.28);
    border-style: dashed;
    inset: 25%;
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
    background: rgba(13, 35, 53, 0.92);
    border: 1px solid rgba(142, 199, 221, 0.55);
    border-radius: 50%;
    box-shadow: 0 0 0 14px rgba(78, 162, 198, 0.06), 0 0 38px rgba(78, 162, 198, 0.16);
    display: flex;
    flex-direction: column;
    height: 128px;
    justify-content: center;
    left: 50%;
    position: absolute;
    text-align: center;
    top: 50%;
    transform: translate(-50%, -50%);
    width: 128px;
    z-index: 2;
}

.scene-core small {
    color: #8EC7DD;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.scene-core strong {
    color: #FFFFFF;
    font-size: 0.96rem;
    font-weight: 500;
    margin-top: 0.25rem;
}

.scene-node {
    background: rgba(7, 24, 39, 0.9);
    border: 1px solid rgba(191, 218, 231, 0.24);
    border-radius: 10px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.22);
    color: #D9E7ED;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    padding: 0.68rem 0.78rem;
    position: absolute;
    text-transform: uppercase;
    z-index: 3;
}

.scene-node::before {
    background: #68D5B3;
    border-radius: 50%;
    content: "";
    display: inline-block;
    height: 5px;
    margin-right: 0.45rem;
    vertical-align: 0.08rem;
    width: 5px;
}

.scene-node.input {
    left: 7%;
    top: 18%;
}

.scene-node.risk {
    right: 6%;
    top: 21%;
}

.scene-node.reason {
    bottom: 18%;
    left: 5%;
}

.scene-node.policy {
    bottom: 15%;
    right: 7%;
}

.scene-sweep {
    border: 1px solid transparent;
    border-top-color: #8EC7DD;
    border-radius: 50%;
    inset: 17%;
    opacity: 0.7;
    position: absolute;
}

.scene-flow {
    inset: 0;
    overflow: visible;
    pointer-events: none;
    position: absolute;
    z-index: 1;
}

.scene-flow path {
    fill: none;
    vector-effect: non-scaling-stroke;
}

.scene-flow-base {
    stroke: rgba(191, 218, 231, 0.15);
    stroke-width: 1;
}

.scene-flow-signal {
    animation: flowSignal 5.8s linear infinite;
    stroke: #68D5B3;
    stroke-dasharray: 3 46;
    stroke-linecap: round;
    stroke-width: 1.7;
}

.scene-caption {
    bottom: 1rem;
    color: #7893A2;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 0.55rem;
    left: 1rem;
    letter-spacing: 0.08em;
    position: absolute;
    text-transform: uppercase;
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
    backdrop-filter: blur(14px);
    background: rgba(244, 249, 248, 0.9);
    border: 1px solid rgba(188, 207, 211, 0.9);
    border-radius: 14px;
    box-shadow: 0 14px 34px rgba(16, 42, 67, 0.11);
    gap: 0.35rem;
    margin: -1.45rem 1.35rem 0;
    padding: 0.42rem;
    overflow-x: auto;
    position: relative;
    z-index: 5;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: 0 !important;
    border-bottom: 0 !important;
    border-radius: 9px !important;
    color: var(--muted) !important;
    flex: 0 0 auto;
    font-size: 0.84rem;
    font-weight: 500;
    overflow: hidden;
    padding: 0.78rem 1rem;
    position: relative;
    transition: background-color 180ms ease, color 180ms ease, transform 180ms ease;
    white-space: nowrap;
}

.stTabs [data-baseweb="tab"]:hover {
    background: #EDF4F7 !important;
    color: var(--atlantic) !important;
    transform: translateY(-1px);
}

.stTabs [aria-selected="true"] {
    background: var(--ink) !important;
    box-shadow: 0 6px 16px rgba(16, 42, 67, 0.18);
    color: var(--ink) !important;
}

.stTabs [aria-selected="true"] p {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #68D5B3, #4EA2C6) !important;
    border-radius: 999px;
    height: 3px !important;
    transition:
        transform 340ms cubic-bezier(0.22, 1, 0.36, 1),
        right 340ms cubic-bezier(0.22, 1, 0.36, 1),
        width 340ms cubic-bezier(0.22, 1, 0.36, 1) !important;
    z-index: 7;
}

.stTabs [data-baseweb="tab-border"] {
    background: rgba(36, 91, 120, 0.12) !important;
    height: 1px !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 1.4rem;
}

.stat-band {
    display: grid;
    grid-template-columns: repeat(var(--stat-count, 3), minmax(0, 1fr));
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-soft);
    margin: 1.2rem 0 1.5rem;
    overflow: hidden;
}

.stat-item {
    min-height: 150px;
    padding: 1.4rem 1.45rem 1.25rem;
    position: relative;
    transition: background-color 180ms ease, transform 180ms ease;
}

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
    background: #F8FBFC;
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
    border-radius: 18px;
    box-shadow: 0 12px 30px rgba(16, 42, 67, 0.065);
    min-height: 220px;
    overflow: hidden;
    padding: 1.35rem 1.35rem 1.2rem;
    position: relative;
    transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
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
    box-shadow: var(--shadow-lift);
    transform: translateY(-4px);
}

.feature-panel:hover::after {
    border-color: var(--atlantic);
    height: 24px;
    width: 24px;
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
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(173, 190, 199, 0.55);
    border-radius: 24px;
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
        radial-gradient(circle at 68% 40%, rgba(78, 162, 198, 0.17), transparent 18rem),
        var(--deep-ocean);
    border-radius: 18px;
    box-shadow: 0 20px 45px rgba(7, 24, 39, 0.18);
    min-height: 620px;
    overflow: hidden;
    padding: 1.5rem;
    position: sticky;
    top: 5.5rem;
}

.story-map::before {
    background-image:
        linear-gradient(rgba(207, 231, 241, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(207, 231, 241, 0.05) 1px, transparent 1px);
    background-size: 30px 30px;
    content: "";
    inset: 0;
    mask-image: linear-gradient(to bottom, black, transparent 92%);
    -webkit-mask-image: linear-gradient(to bottom, black, transparent 92%);
    pointer-events: none;
    position: absolute;
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
    width: calc(100% - 3rem);
    z-index: 1;
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
    stroke-dasharray: 4 8;
    stroke-width: 1.5;
}

.story-progress {
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
    fill: #68D5B3;
    stroke: #DDF8EF;
}

.story-node-label {
    fill: #BFD1DB;
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: 10px;
    letter-spacing: 0.08em;
}

.story-steps {
    display: grid;
    gap: 16vh;
    padding: 8vh 0 12vh;
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
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 18px;
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
}

.risk-panel.low::before {
    background: var(--approval);
}

.risk-panel.high {
    border-left-color: var(--line);
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
    color: var(--ink);
    font-family: "IBM Plex Mono", Consolas, monospace;
    font-size: clamp(2.35rem, 5vw, 3.3rem);
    font-weight: 600;
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

.empty-panel {
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 18px;
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
    background: var(--paper);
    border: 1px solid var(--line);
    border-radius: 18px;
    box-shadow: var(--shadow-soft);
    margin: 1rem 0;
    overflow: hidden;
    position: relative;
}

.policy-ledger::after {
    animation: ledgerBeam 6.5s ease-in-out infinite;
    background: linear-gradient(90deg, transparent, rgba(104, 213, 179, 0.9), transparent);
    content: "";
    height: 1px;
    left: -22%;
    position: absolute;
    top: 0;
    width: 22%;
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

.policy-ledger.review {
    border-left-color: var(--line);
}

.policy-ledger.review::before {
    background: var(--atlantic);
}

.ledger-header {
    align-items: center;
    border-bottom: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.05rem 1.35rem;
}

.ledger-title {
    font-size: 1rem;
    font-weight: 600;
}

.ledger-state {
    align-items: center;
    color: var(--approval);
    display: flex;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}

.ledger-state::before {
    background: currentColor;
    border-radius: 50%;
    box-shadow: 0 0 0 5px rgba(20, 125, 100, 0.09);
    content: "";
    height: 7px;
    margin-right: 0.55rem;
    width: 7px;
}

.policy-ledger.blocked .ledger-state {
    color: var(--block);
}

.policy-ledger.review .ledger-state {
    color: var(--atlantic);
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
    border-radius: 10px !important;
    font-family: "IBM Plex Sans", Arial, sans-serif;
    font-weight: 600 !important;
    min-height: 2.75rem;
    overflow: hidden;
    position: relative;
    transition: box-shadow 180ms ease, transform 180ms ease, background-color 180ms ease;
}

.stButton > button::after {
    background: linear-gradient(105deg, transparent 28%, rgba(255, 255, 255, 0.28) 48%, transparent 68%);
    content: "";
    inset: 0;
    pointer-events: none;
    position: absolute;
    transform: translateX(-125%);
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

.stButton > button[kind="primary"]:hover::after {
    animation: buttonSheen 720ms ease-out;
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
    transition: border-color 200ms ease, box-shadow 200ms ease, transform 200ms ease;
}

@media (hover: hover) {
    .stPlotlyChart:hover {
        border-color: rgba(36, 91, 120, 0.38);
        box-shadow: 0 16px 38px rgba(16, 42, 67, 0.1);
        transform: translateY(-2px);
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

@keyframes heroRise {
    from {
        opacity: 0;
        transform: translateY(18px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes sceneOrbit {
    to {
        transform: rotate(360deg);
    }
}

@keyframes flowSignal {
    to {
        stroke-dashoffset: -196;
    }
}

@keyframes ledgerBeam {
    0%, 18% {
        left: -22%;
        opacity: 0;
    }
    35% {
        opacity: 1;
    }
    62%, 100% {
        left: 104%;
        opacity: 0;
    }
}

@keyframes buttonSheen {
    to {
        transform: translateX(125%);
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

@keyframes signalPulse {
    0%, 100% {
        opacity: 0.55;
        transform: scale(0.98);
    }
    50% {
        opacity: 1;
        transform: scale(1.03);
    }
}

@keyframes viewReveal {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.hero-copy > * {
    animation: heroRise 680ms cubic-bezier(0.2, 0.8, 0.2, 1) both;
}

.hero-copy h1 {
    animation-delay: 90ms;
}

.hero-copy p {
    animation-delay: 170ms;
}

.hero-facts {
    animation-delay: 250ms;
}

.scene-sweep {
    animation: sceneOrbit 18s linear infinite;
}

.scene-core {
    animation: signalPulse 4.5s ease-in-out infinite;
}

@supports (animation-timeline: view()) {
    .stat-band,
    .feature-panel,
    .risk-panel,
    .policy-ledger,
    [data-testid="stAlertContainer"],
    [data-testid="stExpander"],
    .stPlotlyChart,
    [data-testid="stDataFrame"],
    [data-testid="stImage"] {
        animation: viewReveal linear both;
        animation-range: entry 5% cover 24%;
        animation-timeline: view();
    }

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
        transform: translateX(-50%);
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

    .risk-topline,
    .ledger-header {
        align-items: flex-start;
        flex-direction: column;
    }
}

@media (max-width: 600px) {
    .page-masthead {
        border-radius: 18px;
    }

    .hero-copy {
        padding: 2.4rem 1.35rem 1rem;
    }

    .page-masthead h1 {
        font-size: clamp(2.35rem, 12vw, 3.15rem);
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
        max-width: 280px;
    }

    .scene-core {
        height: 96px;
        width: 96px;
    }

    .scene-node {
        font-size: 0.5rem;
        padding: 0.52rem 0.58rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        margin-top: -1rem;
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

@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.001ms !important;
        animation-iteration-count: 1 !important;
        scroll-behavior: auto !important;
        transition-duration: 0.001ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

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
        font=dict(family="IBM Plex Sans, Arial, sans-serif", color="#102A43", size=12),
        title=dict(font=dict(size=15, color="#102A43"), x=0.02, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0)",
            font=dict(color="#5C6F7E"),
            title_font=dict(color="#5C6F7E"),
        ),
        margin=dict(l=58, r=24, t=54, b=56),
        hoverlabel=dict(
            bgcolor="#102A43",
            bordercolor="#102A43",
            font=dict(color="#FFFFFF", family="IBM Plex Sans, Arial, sans-serif"),
        ),
    )
    fig.update_xaxes(
        gridcolor="#E6ECF0",
        linecolor="#B8C5CE",
        tickfont=dict(color="#5C6F7E"),
        title_font=dict(color="#5C6F7E"),
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor="#E6ECF0",
        linecolor="#B8C5CE",
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
        '<div class="ledger-header">'
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

CHURN_COLOR = '#A33A32'
RETAIN_COLOR = '#245B78'

st.markdown(
    f"""
    <div class="page-masthead">
      <div class="hero-copy">
        <div class="hero-kicker">Irish retail banking model</div>
        <h1>
          <span>Know who may leave.</span>
          <span class="hero-accent">Decide with care.</span>
        </h1>
        <p>A working prototype that connects customer risk, model evidence, and a governed response in one review.</p>
        <div class="hero-facts">
          <div class="hero-fact">
            <strong>{len(df_data):,}</strong>
            <span>synthetic customer records</span>
          </div>
          <div class="hero-fact">
            <strong>{len(feature_names)}</strong>
            <span>validated model inputs</span>
          </div>
          <div class="hero-fact">
            <strong>4 tools</strong>
            <span>inside the governed agent</span>
          </div>
        </div>
      </div>
      <div class="hero-scene" aria-label="A visual map from customer signals to a governed retention decision">
        <div class="scene-frame">
          <svg class="scene-flow" viewBox="0 0 410 410" aria-hidden="true">
            <path class="scene-flow-base" d="M78 92 C142 58 268 64 333 104 C305 147 258 170 205 205 C158 237 111 264 80 322 C147 357 260 358 332 326 C295 270 259 232 205 205"></path>
            <path class="scene-flow-signal" d="M78 92 C142 58 268 64 333 104 C305 147 258 170 205 205 C158 237 111 264 80 322 C147 357 260 358 332 326 C295 270 259 232 205 205"></path>
          </svg>
          <div class="scene-axis"></div>
          <div class="scene-axis vertical"></div>
          <div class="scene-sweep"></div>
          <div class="scene-core">
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
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Data explorer",
    "Model performance",
    "SHAP explanations",
    "Risk predictor",
    "Retention agent",
])


with tab1:
    st.header("The Irish banking context")
    st.markdown(
        '<p class="section-note">The project starts with the disruption that led to more than 1.1 million recorded account closures. It uses that period as context for a synthetic study of loyalty and service signals.</p>',
        unsafe_allow_html=True,
    )

    _render_stat_band([
        {
            "label": "Accounts closed",
            "value": "1.17M",
            "note": "The Central Bank recorded 1,167,219 current and deposit account closures at the two exiting banks by the end of June 2023.",
        },
        {
            "label": "Switching difficulty",
            "value": "60%",
            "note": (
                "CCPC research found that six in ten surveyed customers with an open or "
                "recently closed KBC or Ulster current account experienced switching challenges."
            ),
            "tone": "block",
        },
        {
            "label": "Local branch mattered",
            "value": "28%",
            "note": "Among respondents who had switched, 28 percent cited a local branch as a key reason for their choice.",
            "tone": "approval",
        },
    ])

    st.info(
        "As KBC Bank Ireland and Ulster Bank prepared to leave the market, customers who still needed current or "
        "deposit account services had to arrange alternatives. This synthetic study uses that period as context "
        "for loyalty and service signals. It is not a production banking system."
    )
    st.caption(
        "Published context: [Central Bank of Ireland account migration statistics]"
        "(https://www.centralbank.ie/statistics/data-and-analysis/credit-and-banking-statistics/account-migration-statistics) "
        "and [CCPC switching research]"
        "(https://www.ccpc.ie/about-us/advocacy-and-research/research/publication-details/"
        "ccpc-switching-research-%28phase-2%29)."
    )

    st.markdown("#### What the project contains")

    lc1, lc2, lc3 = st.columns([1.25, 1, 1])
    with lc1:
        st.markdown(
            """
            <div class="feature-panel model">
              <div class="feature-index">01 / PREDICT</div>
              <div class="feature-visual" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
              <h4>XGBoost with SMOTEENN</h4>
              <p>The classifier estimates churn probability for each customer. SMOTEENN is applied to training data, then performance is checked on a clean holdout sample.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lc2:
        st.markdown(
            """
            <div class="feature-panel explain">
              <div class="feature-index">02 / EXPLAIN</div>
              <div class="feature-visual" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
              <h4>SHAP explanations</h4>
              <p>Global views show the strongest model signals. A local waterfall explains what moved one customer toward churn or retention.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with lc3:
        st.markdown(
            """
            <div class="feature-panel counter">
              <div class="feature-index">03 / EXPLORE</div>
              <div class="feature-visual" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span></div>
              <h4>DiCE counterfactuals</h4>
              <p>Counterfactual examples show which permitted input changes can produce a model score below the churn threshold.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <section class="system-story" aria-label="How one customer moves through both phases of the project">
          <div class="story-map">
            <div class="story-map-copy">
              <div class="eyebrow">One customer record</div>
              <h3>How the two phases meet</h3>
              <p>This is a process map, not a performance chart. The line follows the real path through the application.</p>
            </div>
            <svg class="story-map-graphic" viewBox="0 0 360 520" aria-hidden="true">
              <path class="story-track" d="M72 44 C240 55 300 100 250 145 C190 198 92 200 100 260 C108 320 300 302 270 365 C245 416 108 426 150 478"></path>
              <path class="story-progress" d="M72 44 C240 55 300 100 250 145 C190 198 92 200 100 260 C108 320 300 302 270 365 C245 416 108 426 150 478"></path>
              <circle class="story-node n1" cx="72" cy="44" r="9"></circle>
              <circle class="story-node n2" cx="250" cy="145" r="9"></circle>
              <circle class="story-node n3" cx="100" cy="260" r="9"></circle>
              <circle class="story-node n4" cx="270" cy="365" r="9"></circle>
              <circle class="story-node core n5" cx="150" cy="478" r="11"></circle>
              <text class="story-node-label" x="92" y="40">PROFILE</text>
              <text class="story-node-label" x="268" y="141">PHASE 1</text>
              <text class="story-node-label" x="120" y="256">EVIDENCE</text>
              <text class="story-node-label" x="290" y="361">AGENT</text>
              <text class="story-node-label" x="170" y="474">REVIEW</text>
            </svg>
          </div>
          <div class="story-steps">
            <article class="story-step">
              <div class="story-step-number">01</div>
              <h4>The selected profile</h4>
              <p>Tab 5 builds one ordered feature vector from the same {len(feature_names)} columns stored with the trained model.</p>
              <div class="story-proof">{len(feature_names)} inputs checked against the saved schema</div>
            </article>
            <article class="story-step">
              <div class="story-step-number">02</div>
              <h4>Phase 1 scores the record</h4>
              <p>The cached XGBoost model runs predict_proba on that feature vector. The selected customer and result are kept together for Tab 6.</p>
              <div class="story-proof">model.predict_proba(feature_vector)[0, 1]</div>
            </article>
            <article class="story-step">
              <div class="story-step-number">03</div>
              <h4>Evidence comes before action</h4>
              <p>A local SHAP explanation shows what moved the score. For higher risk profiles, DiCE can test changes to fields that are allowed to vary.</p>
              <div class="story-proof">TreeExplainer with constrained counterfactuals</div>
            </article>
            <article class="story-step">
              <div class="story-step-number">04</div>
              <h4>The agent checks its proposal</h4>
              <p>Product lookup and segment comparison supply context. The policy checker must run before the recommendation formatter.</p>
              <div class="story-proof">Four local tools with matching call records</div>
            </article>
            <article class="story-step">
              <div class="story-step-number">05</div>
              <h4>A person remains responsible</h4>
              <p>The deterministic gate passes or blocks a proposed action against local project rules. Above the configured 75 percent threshold, the proposal must be marked as requiring advisor review before it can pass. The application neither records a completed review nor contacts customers.</p>
              <div class="story-proof">Review requirement, not review completion</div>
            </article>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        with st.expander("EU AI Act, Article 86"):
            st.markdown(
                "From 2 August 2026, Article 86 provides a right to an explanation for decisions based on the output "
                "of specified Annex III high risk AI systems, except systems listed in point 2, when the decision "
                "produces legal effects or similarly significantly affects a person in a way they consider adverse "
                "to their health, safety, or fundamental rights. "
                "This prototype makes no claim that it falls within that scope. SHAP and DiCE show how model evidence "
                "can be made reviewable. "
                "[Read Article 86 on EUR Lex]"
                "(https://eur-lex.europa.eu/eli/reg/2024/1689/oj#art_86)."
            )
    with exp_col2:
        with st.expander("EBA guidelines on internal governance"):
            st.markdown(
                "The EBA's in-force Guidelines on internal governance under CRD apply to institutions within their "
                "stated scope and address responsibilities, risk management, and internal controls. They do not "
                "prescribe this retention workflow. The checks in this application are engineering choices and have "
                "not been assessed as evidence of regulatory compliance. "
                "[Read the EBA guidelines]"
                "(https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/"
                "internal-governance/guidelines-internal-governance-under-crd)."
            )


with tab2:
    st.header("Churn by customer segment")
    st.markdown(
        '<p class="section-note">A closer look at the 10,000 synthetic customer records used in this study.</p>',
        unsafe_allow_html=True,
    )

    overall_churn = df_data['churn'].mean() * 100
    kbc_mask = df_data['was_kbc_ulster_customer']
    kbc_churn = df_data[kbc_mask]['churn'].mean() * 100
    other_churn = df_data[~kbc_mask]['churn'].mean() * 100
    churn_ratio = kbc_churn / other_churn

    _render_stat_band([
        {
            "label": "Overall churn",
            "value": f"{overall_churn:.1f}%",
            "note": "Measured across all 10,000 customer records.",
        },
        {
            "label": "Former KBC or Ulster",
            "value": f"{kbc_churn:.1f}%",
            "note": f"This is {churn_ratio:.1f} times the rate for other customers.",
            "tone": "block",
        },
        {
            "label": "Other customers",
            "value": f"{other_churn:.1f}%",
            "note": "The comparison rate for customers outside the migration cohort.",
            "tone": "approval",
        },
    ])

    st.info(
        f"Within this synthetic dataset, former KBC and Ulster Bank customer records have a {kbc_churn:.1f}% "
        f"churn rate versus {other_churn:.1f}% for other records ({churn_ratio:.1f}x higher). The gap narrows "
        "as months since switching increases, but it remains visible in the generated data."
    )

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        acct_churn = df_data.groupby('account_type')['churn'].mean().reset_index()
        acct_churn['churn'] *= 100
        fig_acct = px.bar(
            acct_churn,
            x='account_type',
            y='churn',
            labels={'account_type': 'Account type', 'churn': 'Churn rate (%)'},
            title='Churn rate by account type',
            color_discrete_sequence=[CHURN_COLOR]
        )
        _style_plot(fig_acct, height=320)
        fig_acct.update_layout(showlegend=False)
        st.plotly_chart(fig_acct, width="stretch")
        st.caption(
            "In this synthetic dataset, records with only a current or savings account have higher observed churn "
            "than records with a mortgage. This is a pattern in the generated data, not a causal market finding."
        )

        prod_churn = df_data.groupby('num_products')['churn'].mean().reset_index()
        prod_churn['churn'] *= 100
        fig_prod = px.bar(
            prod_churn,
            x='num_products',
            y='churn',
            labels={'num_products': 'Products held', 'churn': 'Churn rate (%)'},
            title='Churn rate by number of products held',
            color_discrete_sequence=[CHURN_COLOR]
        )
        _style_plot(fig_prod, height=320)
        fig_prod.update_layout(showlegend=False)
        st.plotly_chart(fig_prod, width="stretch")
        st.caption(
            "Observed churn falls as the number of products rises in this synthetic dataset. "
            "The chart shows an association created by the study, not the effect of adding a product."
        )

    with c2:
        df_hist = df_data.copy()
        df_hist['churn_label'] = df_hist['churn'].map({0: 'Retained', 1: 'Churned'})
        fig_tenure = px.histogram(
            df_hist,
            x='tenure_months',
            color='churn_label',
            barmode='overlay',
            labels={'tenure_months': 'Tenure (months)', 'count': 'Customers', 'churn_label': 'Status'},
            title='Tenure distribution by churn status',
            color_discrete_map={'Retained': RETAIN_COLOR, 'Churned': CHURN_COLOR}
        )
        _style_plot(fig_tenure, height=320)
        fig_tenure.update_traces(opacity=0.78)
        st.plotly_chart(fig_tenure, width="stretch")
        st.caption(
            "Churned records are concentrated at shorter tenures in this synthetic dataset. "
            "The five year pattern is descriptive here and should not be treated as a production threshold."
        )

        complaint_churn = df_data.groupby('has_complaint_history')['churn'].mean().reset_index()
        complaint_churn['churn'] *= 100
        complaint_churn['has_complaint_history'] = complaint_churn['has_complaint_history'].map(
            {True: 'Complaint on record', False: 'No complaints'}
        )
        fig_complaint = px.bar(
            complaint_churn,
            x='has_complaint_history',
            y='churn',
            labels={'has_complaint_history': 'Complaint status', 'churn': 'Churn rate (%)'},
            title='Churn rate by complaint history',
            color='has_complaint_history',
            color_discrete_map={'Complaint on record': CHURN_COLOR, 'No complaints': RETAIN_COLOR}
        )
        _style_plot(fig_complaint, height=320)
        fig_complaint.update_layout(showlegend=False)
        st.plotly_chart(fig_complaint, width="stretch")
        st.caption(
            "In this synthetic dataset, customers with a complaint record have a higher observed churn rate. "
            "That makes service recovery worth examining."
        )


with tab3:
    st.header("XGBoost performance")
    st.markdown(
        '<p class="section-note">Every score comes from the 20 percent test sample. These records were kept out of training.</p>',
        unsafe_allow_html=True,
    )
    st.success(
        "These scores come from the stratified holdout sample. "
        "XGBoost was benchmarked against Logistic Regression and Random Forest during training."
    )

    _render_stat_band([
        {
            "label": "F1 score",
            "value": f"{xgb_f1:.3f}",
            "note": "Balances precision and recall across the holdout sample.",
        },
        {
            "label": "ROC AUC",
            "value": f"{xgb_roc_auc:.3f}",
            "note": "Measures separation between churned and retained customers.",
            "tone": "approval",
        },
        {
            "label": "Average precision",
            "value": f"{xgb_average_precision:.3f}",
            "note": "Summarises precision and recall where churn is less common.",
        },
    ])

    st.divider()

    ch1, ch2, ch3 = st.columns(3)

    with ch1:
        cm = xgb_confusion_matrix
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Predicted: retained', 'Predicted: churned'],
            y=['Actual: retained', 'Actual: churned'],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            color_continuous_scale='Blues',
            title='Confusion matrix'
        )
        _style_plot(fig_cm, height=360)
        st.plotly_chart(fig_cm, width="stretch")
        st.caption(
            f"{int(cm[1, 1])} of {int(cm[1].sum())} actual churn records were correctly identified. "
            f"The {int(cm[1, 0])} false negatives are missed churn records in this synthetic holdout sample."
        )

    with ch2:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = px.line(
            x=fpr,
            y=tpr,
            labels={'x': 'False positive rate', 'y': 'True positive rate'},
            title=f'ROC curve  (AUC = {xgb_roc_auc:.3f})'
        )
        fig_roc.add_shape(type='line', line=dict(dash='dash', color='gray'), x0=0, x1=1, y0=0, y1=1)
        _style_plot(fig_roc, height=360)
        st.plotly_chart(fig_roc, width="stretch")
        st.caption(
            "The curve close to the upper left corner shows strong separation between classes. "
            "The dashed diagonal is random chance (AUC = 0.50)."
        )

    with ch3:
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        fig_pr = px.line(
            x=recall,
            y=precision,
            labels={'x': 'Recall', 'y': 'Precision'},
            title=f'Precision recall curve  (AP = {xgb_average_precision:.3f})'
        )
        class_prevalence = float(y_test.mean())
        fig_pr.add_shape(
            type='line',
            line=dict(dash='dash', color='gray'),
            x0=0,
            x1=1,
            y0=class_prevalence,
            y1=class_prevalence,
        )
        _style_plot(fig_pr, height=360)
        st.plotly_chart(fig_pr, width="stretch")
        st.caption(
            f"The dashed baseline is the class prevalence ({class_prevalence:.0%}). "
            "The model stays above this line across most operating points."
        )


with tab4:
    st.header("Global feature importance")
    st.markdown(
        f'<p class="section-note">These views show which customer details had the strongest effect across the {len(X_test):,} record holdout sample.</p>',
        unsafe_allow_html=True,
    )

    st.info(
        f"SHAP gives each feature a contribution score for every prediction. Features are ranked by their "
        f"average impact across the {len(X_test):,} records kept out of training. "
        "Positive values move a prediction toward churn. Negative values move it toward retention."
    )

    st.markdown("##### Five strongest features by mean absolute SHAP value")
    rank_col1, rank_col2 = st.columns(2)
    with rank_col1:
        st.markdown("""
| Rank | Feature | SHAP |
|:---:|:---|:---:|
| 1 | `num_products` | 2.841 |
| 2 | `months_since_switching` | 1.028 |
| 3 | `has_direct_debits` | 0.883 |
| 4 | `tenure_months` | 0.838 |
| 5 | `has_savings_goal` | 0.529 |
        """)
    with rank_col2:
        st.markdown("""
**Reading the ranking**

The fitted model gives its largest average SHAP effects to product count and time since switching. In this synthetic holdout sample, those fields moved predictions more than the other inputs. This ranking describes the model and does not establish a causal effect in the Irish market.

The next three inputs are direct debits, tenure, and a savings goal. Their positions show how the fitted model used the generated data, not how every real customer would behave.
        """)

    shap_col1, shap_col2 = st.columns(2)

    with shap_col1:
        st.markdown("**Beeswarm plot and feature interactions**")
        beeswarm_path = os.path.join('assets', 'shap_summary_plot.png')
        if os.path.exists(beeswarm_path):
            st.image(beeswarm_path, width="stretch")
            st.caption(
                "Each dot represents one customer. Red shows a higher feature value and blue shows a lower one. "
                "Points on the right move the raw model output toward the churn class. "
                "A low product count is the strongest single signal."
            )
        else:
            st.warning("Beeswarm plot not found. Run models/train_model.py to generate it.")

    with shap_col2:
        st.markdown("**Bar plot and mean absolute impact**")
        bar_path = os.path.join('assets', 'shap_bar_plot.png')
        if os.path.exists(bar_path):
            st.image(bar_path, width="stretch")
            st.caption(
                f"Features ranked by their average impact across the {len(X_test):,} record holdout sample. "
                "A longer bar means a larger average absolute contribution to the model's raw output. "
                "The SHAP values are not probability percentage points."
            )
        else:
            st.warning("Bar plot not found. Run models/train_model.py to generate it.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        "Article 86 is due to apply from 2 August 2026. This prototype makes no claim that it falls within the "
        "Article's scope. Where its conditions are met, an affected person can obtain a clear and meaningful "
        "explanation of the AI system's role and the main elements of the decision. The SHAP views here demonstrate "
        "model evidence that a reviewer could inspect."
    )


with tab5:
    st.header("Customer risk assessment")
    st.markdown(
        '<p class="section-note">Build a customer profile to see its churn probability, the main model signals, and candidate counterfactual examples.</p>',
        unsafe_allow_html=True,
    )

    col_input, col_output = st.columns([1, 1.5])

    with col_input:

        st.markdown("**Customer profile**")
        customer_reference = st.text_input(
            "Customer reference",
            value="TAB5_CUSTOMER",
            help="Synthetic identifier carried unchanged into the Retention Agent.",
        )
        r1a, r1b = st.columns(2)
        with r1a:
            age = st.slider("Age", 18, 75, 42)
        with r1b:
            tenure_months = st.slider("Tenure (months)", 1, 180, 24)

        st.markdown("**Account details**")
        r2a, r2b = st.columns(2)
        with r2a:
            account_type = st.selectbox(
                "Account type",
                options=['Current Account', 'Savings Account', 'Current + Savings', 'Current + Mortgage']
            )
        with r2b:
            monthly_balance_eur = st.slider("Monthly balance (EUR)", 0, 50000, 2500)

        r3a, r3b = st.columns(2)
        with r3a:
            num_products = st.slider("Products held", 1, 5, 2)
        with r3b:
            credit_score_band = st.selectbox("Credit score band", options=['Low', 'Medium', 'High'])

        r4a, r4b = st.columns(2)
        with r4a:
            monthly_transaction_count = st.slider("Monthly transactions", 5, 200, 45)
        with r4b:
            monthly_transaction_amount_eur = st.slider("Monthly spend (EUR)", 100, 8000, 1200)

        r5a, r5b = st.columns(2)
        with r5a:
            has_direct_debits = st.checkbox("Has direct debits", value=True)
        with r5b:
            direct_debit_count = st.slider(
                "Direct debit count",
                0, 15, 4 if has_direct_debits else 0,
                disabled=not has_direct_debits
            )

        st.markdown("**Switching history**")
        was_kbc_ulster_customer = st.checkbox("Former KBC Bank Ireland or Ulster Bank customer", value=False)
        if was_kbc_ulster_customer:
            sw1, sw2 = st.columns(2)
            with sw1:
                months_since_switching = st.slider("Months since switching", 1, 36, 12)
            with sw2:
                experienced_switching_difficulty = st.checkbox("Experienced switching difficulties")
        else:
            months_since_switching = 0
            experienced_switching_difficulty = False

        st.markdown("**Engagement signals**")
        e1, e2 = st.columns(2)
        with e1:
            uses_digital_bank_secondary = st.checkbox("Uses Revolut / N26 as secondary bank")
            has_mortgage = st.checkbox("Has mortgage with this bank")
            if has_mortgage and account_type != 'Current + Mortgage':
                account_type = 'Current + Mortgage'
        with e2:
            has_complaint_history = st.checkbox("Has complaint on record")
            has_savings_goal = st.checkbox("Has active savings goal")

        e3, e4 = st.columns(2)
        with e3:
            branch_visits_monthly = st.slider("Branch visits per month", 0, 8, 1)
        with e4:
            customer_service_calls_6months = st.slider("Service calls (last 6 months)", 0, 12, 1)

        predict_btn = st.button("Predict churn risk", type="primary", width="stretch")

    with col_output:

        if predict_btn:
            raw_profile = {
                'age': age,
                'tenure_months': tenure_months,
                'account_type': account_type,
                'monthly_balance_eur': float(monthly_balance_eur),
                'num_products': num_products,
                'monthly_transaction_count': monthly_transaction_count,
                'monthly_transaction_amount_eur': float(monthly_transaction_amount_eur),
                'has_direct_debits': has_direct_debits,
                'direct_debit_count': int(direct_debit_count) if has_direct_debits else 0,
                'uses_digital_bank_secondary': uses_digital_bank_secondary,
                'was_kbc_ulster_customer': was_kbc_ulster_customer,
                'months_since_switching': months_since_switching,
                'experienced_switching_difficulty': experienced_switching_difficulty,
                'branch_visits_monthly': branch_visits_monthly,
                'customer_service_calls_6months': customer_service_calls_6months,
                'has_complaint_history': has_complaint_history,
                'credit_score_band': credit_score_band,
                'has_mortgage': has_mortgage,
                'has_savings_goal': has_savings_goal,
            }
            held_products = []
            if "Current" in account_type:
                held_products.append("current_account")
            if "Savings" in account_type:
                held_products.append("savings_account")
            if has_mortgage:
                held_products.append("mortgage")
            phase1_customer = {
                "customer_id": customer_reference.strip() or "TAB5_CUSTOMER",
                "profile": raw_profile,
                "held_products": sorted(set(held_products)),
                "governance": {
                    "in_arrears": False,
                    "vulnerable_customer": False,
                },
                "governance_note": (
                    "Tab 5 supplies model features only. The synthetic governance "
                    "overlay defaults to no arrears and not vulnerable."
                ),
                "counterfactuals": [],
                "churn_drivers": [],
            }
            try:
                phase1_prediction = predict_customer_churn_risk(
                    phase1_customer,
                    phase1_runtime=phase1_runtime,
                )
            except Phase1SchemaError as exc:
                st.error(f"Phase 1 feature schema validation failed: {exc}")
                st.stop()
            phase1_customer["churn_probability"] = phase1_prediction[
                "churn_probability"
            ]
            phase1_customer["phase1_prediction"] = phase1_prediction
            input_df = phase1_runtime.prepare_feature_vector(phase1_customer)
            churn_prob = phase1_prediction["churn_probability"]

            if churn_prob < 0.30:
                risk_label = "Lower score band"
                risk_msg = "The fitted model assigns a lower churn probability to this synthetic profile."
                risk_class = "low"
            elif churn_prob < 0.60:
                risk_label = "Middle score band"
                risk_msg = "The fitted model assigns a middle range churn probability to this synthetic profile."
                risk_class = ""
            else:
                risk_label = "Higher score band"
                risk_msg = (
                    "The fitted model assigns a higher churn probability to this synthetic profile. "
                    "Treat the score as decision support, not a customer decision."
                )
                risk_class = "high"

            st.markdown(f"""
<div class="risk-panel {risk_class}">
  <div class="risk-topline">
    <div>
      <div class="stat-label">Churn probability</div>
      <div class="risk-value">{churn_prob*100:.1f}%</div>
    </div>
    <div class="status-label {risk_class}">{risk_label}</div>
  </div>
  <p>{risk_msg}</p>
</div>""", unsafe_allow_html=True)
            st.progress(float(churn_prob))
            st.caption(
                "These locally configured display bands use cutoffs of 30 and 60 percent. "
                "They are not calibrated or externally validated risk categories."
            )

            st.divider()

            st.markdown("##### Local SHAP explanation")
            explainer = shap.TreeExplainer(xgb_model)
            shap_val = explainer(input_df)
            shap_values = np.asarray(shap_val.values[0], dtype=float)
            top_driver_indices = np.argsort(np.abs(shap_values))[::-1][:5]
            phase1_customer["churn_drivers"] = [
                {
                    "feature": feature_names[index],
                    "value": raw_profile[feature_names[index]],
                    "shap_value": float(shap_values[index]),
                    "direction": (
                        "increases_churn"
                        if shap_values[index] >= 0
                        else "decreases_churn"
                    ),
                }
                for index in top_driver_indices
            ]
            st.session_state["phase1_selected_customer"] = phase1_customer
            st.session_state.pop("retention_live_result", None)
            fig, ax = plt.subplots(figsize=(5, 3.5))
            shap.plots.waterfall(shap_val[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Red bars pushed this customer toward churn. Blue bars pulled toward retention. "
                "The longer the bar, the stronger the effect."
            )

            if churn_prob > 0.50:
                st.divider()

                st.markdown("##### Which input changes produce a different model score?")
                st.caption(
                    "These are candidate model scenarios, not evidence that changing a customer circumstance "
                    "would prevent churn. Any real use would need separate suitability and governance review."
                )

                with st.spinner("Generating counterfactual scenarios..."):
                    try:
                        # lock features that cannot realistically change for an existing customer
                        cf = dice_explainer.generate_counterfactuals(
                            input_df,
                            total_CFs=3,
                            desired_class=0,
                            features_to_vary=[
                                c for c in feature_names
                                if c not in ['age', 'was_kbc_ulster_customer', 'experienced_switching_difficulty']
                            ]
                        )

                        if cf is not None and len(cf.cf_examples_list) > 0:
                            cf_df = cf.cf_examples_list[0].final_cfs_df
                            orig_row = cf_df.iloc[0]
                            cf_rows = cf_df.iloc[1:]

                            changes = []
                            for col in input_df.columns:
                                orig_val = orig_row[col]
                                cf_vals = cf_rows[col].unique()

                                if not np.all(cf_vals == orig_val):
                                    cf_disp_list = []
                                    for val in cf_vals:
                                        if val == orig_val:
                                            continue
                                        if col in encoders:
                                            cf_disp_list.append(
                                                str(encoders[col].inverse_transform([int(val)])[0])
                                            )
                                        elif col in ['has_direct_debits', 'uses_digital_bank_secondary',
                                                     'has_complaint_history', 'has_mortgage', 'has_savings_goal']:
                                            cf_disp_list.append("Yes" if val == 1 else "No")
                                        elif col in ['monthly_balance_eur', 'monthly_transaction_amount_eur']:
                                            cf_disp_list.append(f"€{val:,.2f}")
                                        else:
                                            cf_disp_list.append(str(int(val)))

                                    if col in encoders:
                                        orig_disp = encoders[col].inverse_transform([int(orig_val)])[0]
                                    elif col in ['has_direct_debits', 'uses_digital_bank_secondary',
                                                 'has_complaint_history', 'has_mortgage', 'has_savings_goal']:
                                        orig_disp = "Yes" if orig_val == 1 else "No"
                                    elif col in ['monthly_balance_eur', 'monthly_transaction_amount_eur']:
                                        orig_disp = f"€{orig_val:,.2f}"
                                    else:
                                        orig_disp = str(int(orig_val))

                                    changes.append({
                                        'Feature': col.replace('_', ' ').title(),
                                        'Current Value': orig_disp,
                                        'Candidate Input': " or ".join(cf_disp_list)
                                    })

                            if changes:
                                st.dataframe(pd.DataFrame(changes), width="stretch", hide_index=True)
                            else:
                                st.write("No distinct changes identified in the counterfactuals.")
                        else:
                            st.write("Could not generate counterfactual suggestions for this profile.")
                    except Exception as e:
                        st.error(f"Error generating suggestions: {e}")

            else:
                st.divider()
                st.success("This customer is below the churn threshold. No counterfactual suggestions needed.")

        else:
            st.markdown("""
<div class="empty-panel">
  <strong>No prediction yet</strong>
  <p>Complete the customer profile, then choose <b>Predict churn risk</b>. The probability, explanation, and counterfactual examples will appear here.</p>
</div>""", unsafe_allow_html=True)


with tab6:
    st.header("Retention agent")
    st.markdown(
        '<p class="section-note">'
        "Inspect how a proposed retention action moves through four tools and a "
        "deterministic policy gate. Blocked outcomes are governed results, not errors."
        "</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "This is a synthetic demonstration. The customer records, governance flags, offers, "
        "and recommendations are not real banking decisions."
    )
    st.caption(
        "ARR-001, HOLD-002, HUM-003, and VUL-004 are local demonstration rules. "
        "They are not legal findings and do not reproduce the Central Bank Consumer "
        "Protection Code or a bank's own eligibility controls."
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

        tab5_customer = st.session_state.get("phase1_selected_customer")
        using_tab5_customer = isinstance(tab5_customer, dict)
        trace_is_recorded = False
        if using_tab5_customer:
            customer = tab5_customer
            recommendation = None
            trace = []
            st.success(
                "This run uses the customer created in the risk predictor: "
                f"`{customer['customer_id']}`"
            )
            st.caption(
                "The risk predictor stored this object after a validated Phase 1 model call. "
                "The same object is passed to the agent."
            )
        else:
            trace_is_recorded = True
            selected_title = st.selectbox(
                "Select a recorded governed scenario",
                options=[record["title"] for record in demo_records],
                format_func=_natural_prose,
            )
            selected_demo = next(
                record for record in demo_records if record["title"] == selected_title
            )
            customer = selected_demo["customer"]
            recommendation = selected_demo["recommendation"]
            trace = selected_demo["trace"]
            st.info(
                "No Tab 5 prediction is in this session. Showing a recorded scenario "
                "whose probability is verified against the trained Phase 1 model."
            )

        if api_key_available:
            st.success(
                f"A Groq key is configured. Groq will validate the key and model access "
                f"when the live run starts. {live_runs_remaining} of "
                f"{SESSION_RUN_CAP} session runs remain."
            )
            st.caption(
                "The app reads the key from deployment secrets or the local environment. "
                "The interface does not accept or display it."
            )
        elif live_runs_remaining <= 0:
            st.caption(
                f"Session cap reached ({SESSION_RUN_CAP}/{SESSION_RUN_CAP} live runs used)."
            )
        elif using_tab5_customer:
            st.warning(
                "The live Phase 1 customer is ready, but GROQ_API_KEY is missing or does "
                "not pass the local format check. "
                "No retention agent output has been generated for this customer."
            )
        else:
            st.warning(
                "GROQ_API_KEY is missing or does not pass the local format check on this "
                "deployment, so the recorded fallback is shown without making an API request."
            )

        if api_key_available:
            st.info(
                f"**Groq runtime.** Model: `{MODEL_NAME}` · "
                f"Calls per run: {MAX_LIVE_API_CALLS} · "
                f"Tokens per call: {MAX_TOKENS} · "
                f"Loop turns: {MAX_LOOP_TURNS} · "
                f"Local daily request budget remaining: "
                f"{quota_snapshot['daily_requests_remaining']}"
            )
            st.caption(
                "This local process counter does not read Groq account usage or requests "
                "made by another running instance."
            )
            if st.button(
                "Run live governed recommendation",
                disabled=(
                    live_runs_remaining <= 0
                    or quota_snapshot["daily_requests_remaining"] <= 0
                ),
                type="primary",
            ):
                if live_runs_remaining <= 0:
                    st.error("Session demo limit reached (5 runs), please try again later.")
                else:
                    try:
                        GLOBAL_REQUEST_QUOTA.ensure_run_available()
                        reserve_session_run(st.session_state)
                        live_client = create_live_client(api_key=groq_api_key)
                        with st.spinner("The agent is checking the customer and policy rules..."):
                            live_result = run_retention_agent(
                                customer,
                                client=live_client,
                                phase1_runtime=phase1_runtime,
                            )
                        st.session_state["retention_live_result"] = {
                            "customer_id": customer["customer_id"],
                            "payload": live_result,
                        }
                    except RateLimitSafetyError as exc:
                        st.error(str(exc))
                    except Exception as exc:
                        status_code = getattr(exc, "status_code", None)
                        error_text = str(exc).lower()
                        if status_code == 401 or "invalid_api_key" in error_text:
                            st.error(
                                "Live Groq authentication failed. The deployment owner "
                                "must replace GROQ_API_KEY in Streamlit Secrets with a "
                                "valid Groq API key."
                            )
                        else:
                            st.error(f"Live run stopped safely: {exc}")

            stored_live_result = st.session_state.get("retention_live_result")
            if (
                stored_live_result
                and stored_live_result.get("customer_id") == customer["customer_id"]
            ):
                trace_is_recorded = False
                customer = stored_live_result["payload"]["customer"]
                recommendation = stored_live_result["payload"]["recommendation"]
                trace = stored_live_result["payload"]["trace"]
                st.caption(
                    "Showing the live Groq result for the same Tab 5 customer object. "
                    "Phase 1 risk was recomputed inside the agent run."
                )

        st.divider()
        profile_col, driver_col = st.columns([1, 1.15])
        with profile_col:
            st.subheader("Synthetic customer profile")
            metric_left, metric_right = st.columns(2)
            metric_left.metric("Customer", customer["customer_id"])
            metric_right.metric(
                "Churn probability",
                f'{customer["churn_probability"]:.1%}',
            )
            phase1_evidence = customer.get("phase1_prediction", {})
            if phase1_evidence:
                st.caption(
                    "Runtime source: "
                    f'`{phase1_evidence.get("prediction_method")}` · '
                    f'{len(phase1_evidence.get("feature_columns", []))} ordered features'
                )
            profile_rows = [
                {
                    "Field": key.replace("_", " ").title(),
                    "Value": str(value),
                }
                for key, value in customer["profile"].items()
            ]
            st.dataframe(
                pd.DataFrame(profile_rows),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Held products: "
                + ", ".join(
                    _display_identifier(product)
                    for product in customer.get("held_products", [])
                )
            )
            governance = customer.get("governance", {})
            st.caption(
                "Synthetic governance overlay. "
                f'In arrears: {governance.get("in_arrears", False)}. '
                "Vulnerable customer: "
                f'{governance.get("vulnerable_customer", False)}'
            )

        with driver_col:
            st.subheader("Phase 1 churn drivers")
            driver_rows = []
            for driver in customer.get("churn_drivers", []):
                driver_rows.append(
                    {
                        "Feature": driver["feature"].replace("_", " ").title(),
                        "Customer value": str(driver["value"]),
                        "SHAP value": round(driver["shap_value"], 4),
                        "Direction": driver["direction"].replace("_", " "),
                    }
                )
            st.dataframe(
                pd.DataFrame(driver_rows),
                width="stretch",
                hide_index=True,
            )
            st.info(_natural_prose(customer["governance_note"]))

        if not trace or recommendation is None:
            st.divider()
            st.info(
                "The Phase 1 customer is ready for the retention agent. "
                "Configure Groq and choose **Run live governed recommendation** to create a governed "
                "recommendation and trace for this exact customer."
            )
            st.stop()

        st.divider()
        st.subheader("Reasoning and governance trace")
        if trace_is_recorded:
            st.caption(
                "This is a recorded, zero request scripted replay. Its Phase 1 probability came from "
                "the local trained model, but the reasoning text is not a live Groq response. "
                "Open a raw view only when you need the full payload."
            )
        else:
            st.caption(
                "This trace came from the live Groq run shown above. Read the plain language step "
                "titles first, then open a raw view when you need the full payload."
            )
        visible_steps = st.slider(
            "Replay through step",
            min_value=1,
            max_value=len(trace),
            value=len(trace),
        )
        for event in trace[:visible_steps]:
            event_type = event["type"]
            content = event["content"]
            step_label = f'Step {event["step"]}'

            if event_type == "model_thought":
                with st.expander(f"{step_label}: Review the available evidence", expanded=True):
                    if trace_is_recorded:
                        st.write(
                            "The recorded script describes how the evidence would be considered "
                            "at this step."
                        )
                    else:
                        st.write(
                            "The model considered the customer evidence before choosing the next step."
                        )
                    with st.expander("View raw analysis"):
                        st.write(content.get("text", content))

            elif event_type == "tool_call":
                name = content.get("name", "unknown")
                inp = content.get("input", {})
                summary = _trace_call_summary(name, inp)
                with st.expander(
                    f"{step_label}: {summary}",
                    expanded=True,
                ):
                    with st.expander("View raw payload"):
                        st.json(content)

            elif event_type == "tool_result":
                is_error = content.get("is_error", False)
                name = content.get("name", "unknown")
                if is_error:
                    with st.expander(
                        f"{step_label}: Tool error in {_display_identifier(name)}",
                        expanded=True,
                    ):
                        st.error(content.get("result", content))
                else:
                    result = content.get("result", {})
                    summary = _trace_result_summary(name, result)
                    with st.expander(
                        f"{step_label}: {summary}",
                        expanded=True,
                    ):
                        with st.expander("View raw result"):
                            st.json(content)

            elif event_type == "gate_check":
                passed = content.get("passed")
                action_id = content.get("action_id", "action")
                failed_ids = content.get("failed_rule_ids", [])
                if passed:
                    st.success(
                        f"{step_label}: Local policy gate passed for {_display_identifier(action_id)}"
                    )
                else:
                    st.error(
                        f"{step_label}: Local policy gate blocked {_display_identifier(action_id)}"
                        + (f". Failed rules: {', '.join(failed_ids)}" if failed_ids else "")
                    )
                for rule_result in content.get("rule_results", []):
                    rule_state = "Passed" if rule_result.get("passed") else "Blocked"
                    st.caption(
                        f"{rule_state}. {rule_result['rule_id']}: "
                        f"{_natural_prose(rule_result['reason'])}"
                    )
                with st.expander("View full gate decision"):
                    st.json(content)

            elif event_type == "final_output":
                verdict = content.get("checker_verdict", "blocked")
                action = content.get("action", "?")
                confidence = content.get("confidence", 0)
                flags = content.get("regulatory_flags", [])
                if verdict == "approved":
                    st.success(
                        f"{step_label}: Output passed the local gate. "
                        f"Action: {_display_identifier(action)}. "
                        f"Agent confidence: {confidence:.0%}, not calibrated."
                    )
                else:
                    st.error(
                        f"{step_label}: Output blocked by the local gate. No recommendation was issued."
                    )
                if flags:
                    st.caption("Policy records: " + ", ".join(flags))
                with st.expander("View structured output"):
                    st.json(content)

        st.divider()
        st.subheader("Policy checked output")
        _render_policy_ledger(recommendation)
