"""Regression coverage for the Atlantic Ledger Ledger Gate identity system.

These tests deliberately treat ``atlantic-ledger-brand.json`` as the only
editable source of logo geometry.  Everything else is a committed generated
artifact or a consumer of that contract.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRAND_DIR = PROJECT_ROOT / "assets" / "brand"
MANIFEST_PATH = BRAND_DIR / "atlantic-ledger-brand.json"
WEB_ROOT = PROJECT_ROOT / "web"

EXPECTED_PATHS = {
    "standard": {
        "diagonal": "M24 6 H39 L20 58 H6 Z",
        "gateBody": "M36 6 H46 V29 H40 V41 H46 V48 H58 V58 H36 Z",
        "crossbar": "M15 30 H40 V40 H15 Z",
    },
    "micro": {
        "diagonal": "M24 4 H40 L20 60 H4 Z",
        "gateBody": "M36 4 H48 V28 H40 V44 H48 V48 H60 V60 H36 Z",
        "crossbar": "M14 28 H40 V40 H14 Z",
    },
}

EXPECTED_EXPORTS = {
    "standardSvg": "assets/brand/atlantic-ledger-mark.svg",
    "microSvg": "assets/brand/atlantic-ledger-mark-micro.svg",
    "monoSvg": "assets/brand/atlantic-ledger-mark-mono.svg",
    "favicon128": "assets/brand/atlantic-ledger-favicon-128.png",
    "comparison1x": "assets/brand/atlantic-ledger-size-comparison-1x.svg",
    "comparison2x": "assets/brand/atlantic-ledger-size-comparison-2x.svg",
    "nextIcon": "web/src/app/icon.svg",
    "appleIcon180": "web/src/app/apple-icon.png",
    "typescript": "web/src/components/brand-geometry.generated.ts",
}

LEGACY_BRAND_FILENAMES = {
    "irish_banking_churn_mark.svg",
    "irish_banking_churn_mark_mono.svg",
    "irish_banking_churn_favicon.png",
}


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _svg_root(path: Path) -> ElementTree.Element:
    return ElementTree.fromstring(path.read_text(encoding="utf-8"))


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = path.read_bytes()
    assert payload[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert payload[12:16] == b"IHDR", f"{path} has no leading IHDR chunk"
    return struct.unpack(">II", payload[16:24])


def _generated_text_paths(manifest: dict) -> list[Path]:
    return [
        PROJECT_ROOT / manifest["exports"][key]
        for key in (
            "standardSvg",
            "microSvg",
            "monoSvg",
            "comparison1x",
            "comparison2x",
            "nextIcon",
            "typescript",
        )
    ]


def test_manifest_is_the_canonical_ledger_gate_contract() -> None:
    manifest = _manifest()

    assert set(manifest) == {
        "brand",
        "viewBox",
        "clearSpace",
        "colors",
        "marks",
        "tones",
        "exports",
    }
    assert manifest["brand"] == {
        "name": "Atlantic Ledger",
        "markName": "Ledger Gate",
        "label": "Atlantic Ledger logo",
    }
    assert manifest["viewBox"] == "0 0 64 64"
    assert manifest["clearSpace"] == 10
    assert manifest["colors"] == {
        "atlanticInk": "#071827",
        "ledgerPaper": "#F4F1E8",
        "atlanticBlue": "#245B78",
    }
    assert manifest["marks"]["standard"]["minSize"] == 24
    assert manifest["marks"]["micro"]["minSize"] == 16
    assert manifest["marks"]["micro"]["maxSize"] == 23
    assert manifest["exports"] == EXPECTED_EXPORTS


def test_standard_and_micro_paths_are_exact_and_within_the_64px_master() -> None:
    manifest = _manifest()

    for variant, expected in EXPECTED_PATHS.items():
        paths = manifest["marks"][variant]["paths"]
        assert paths == expected

        for part, path_data in paths.items():
            coordinates = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path_data)]
            assert coordinates, f"{variant}.{part} has no coordinates"
            assert min(coordinates) >= 0, f"{variant}.{part} escapes the viewBox"
            assert max(coordinates) <= 64, f"{variant}.{part} escapes the viewBox"


def test_tones_keep_identity_neutral_and_micro_assets_monochrome() -> None:
    tones = _manifest()["tones"]

    assert tones == {
        "ink": {
            "minSize": 16,
            "diagonal": "#071827",
            "gateBody": "#071827",
            "crossbar": "#071827",
        },
        "reverse": {
            "minSize": 16,
            "diagonal": "#F4F1E8",
            "gateBody": "#F4F1E8",
            "crossbar": "#F4F1E8",
        },
        "blue": {
            "minSize": 24,
            "diagonal": "#245B78",
            "gateBody": "#245B78",
            "crossbar": "#245B78",
        },
        "duotone": {
            "minSize": 32,
            "diagonal": "#245B78",
            "gateBody": "#071827",
            "crossbar": "#245B78",
        },
    }

    semantic_decision_colors = {"#147D64", "#A66F20", "#A33A32"}
    used_logo_colors = {
        value
        for tone in tones.values()
        for key, value in tone.items()
        if key != "minSize"
    }
    assert used_logo_colors.isdisjoint(semantic_decision_colors)


def test_all_declared_assets_exist_and_carry_the_manifest_parity_marker() -> None:
    manifest = _manifest()
    normalized_manifest = (
        MANIFEST_PATH.read_text(encoding="utf-8")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .encode("utf-8")
    )
    manifest_sha = hashlib.sha256(normalized_manifest).hexdigest()
    parity_marker = (
        f"Generated from {MANIFEST_PATH.name}; sha256:{manifest_sha}"
    )

    for relative_path in manifest["exports"].values():
        generated_path = PROJECT_ROOT / relative_path
        assert generated_path.is_file(), f"Missing generated brand asset: {relative_path}"
        assert generated_path.stat().st_size > 0

    for generated_path in _generated_text_paths(manifest):
        source = generated_path.read_text(encoding="utf-8")
        assert parity_marker in source.splitlines()[0]

    assert _png_dimensions(PROJECT_ROOT / manifest["exports"]["favicon128"]) == (
        128,
        128,
    )
    assert _png_dimensions(PROJECT_ROOT / manifest["exports"]["appleIcon180"]) == (
        180,
        180,
    )

    comparison_1x = _svg_root(PROJECT_ROOT / manifest["exports"]["comparison1x"])
    comparison_2x = _svg_root(PROJECT_ROOT / manifest["exports"]["comparison2x"])
    assert comparison_1x.attrib["viewBox"] == "0 0 1080 1400"
    assert comparison_1x.attrib["width"] == "1080"
    assert comparison_1x.attrib["height"] == "1400"
    assert comparison_2x.attrib["viewBox"] == "0 0 1080 1400"
    assert comparison_2x.attrib["width"] == "2160"
    assert comparison_2x.attrib["height"] == "2800"


@pytest.mark.parametrize(
    "export_key",
    ["standardSvg", "microSvg", "monoSvg", "nextIcon"],
)
def test_generated_svgs_use_filled_accessible_64px_geometry(export_key: str) -> None:
    manifest = _manifest()
    svg_path = PROJECT_ROOT / manifest["exports"][export_key]
    root = _svg_root(svg_path)

    assert root.attrib["viewBox"] == "0 0 64 64"
    assert root.attrib["role"] == "img"
    assert root.attrib["aria-label"] == manifest["brand"]["label"]
    assert root.attrib["focusable"] == "false"
    assert "aria-labelledby" not in root.attrib
    assert "id" not in root.attrib

    forbidden_elements = {"circle", "linearGradient", "radialGradient", "filter", "mask"}
    elements = list(root.iter())
    assert all("id" not in node.attrib for node in elements)
    assert forbidden_elements.isdisjoint({_xml_local_name(node.tag) for node in elements})
    assert all("stroke" not in node.attrib for node in elements)

    paths = [node for node in elements if _xml_local_name(node.tag) == "path"]
    assert len(paths) == 3
    assert all(node.attrib.get("fill") not in {None, "", "none"} for node in paths)
    for node in paths:
        coordinates = [
            float(value)
            for value in re.findall(r"-?\d+(?:\.\d+)?", node.attrib["d"])
        ]
        assert min(coordinates) >= 0
        assert max(coordinates) <= 64

    lowered = svg_path.read_text(encoding="utf-8").lower()
    for forbidden_fragment in ("brand-check", "checkmark", "gradient", "filter", "mask"):
        assert forbidden_fragment not in lowered


def test_generated_typescript_matches_every_manifest_path_and_public_variant() -> None:
    manifest = _manifest()
    generated = (
        PROJECT_ROOT / manifest["exports"]["typescript"]
    ).read_text(encoding="utf-8")

    for paths in EXPECTED_PATHS.values():
        for path_data in paths.values():
            assert path_data in generated

    for public_export in (
        "BRAND_LABEL",
        "BRAND_VIEW_BOX",
        "BRAND_CLEAR_SPACE",
        "BRAND_COLORS",
        "BRAND_MARKS",
        "BRAND_TONES",
        "BrandVariant",
        "BrandTone",
        "BrandPart",
    ):
        assert re.search(rf"\b{public_export}\b", generated)


def test_streamlit_and_next_consumers_use_the_new_accessible_brand_contract() -> None:
    manifest = _manifest()
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    component_source = (
        WEB_ROOT / "src" / "components" / "brand-mark.tsx"
    ).read_text(encoding="utf-8")
    lockup_source = (
        WEB_ROOT / "src" / "components" / "brand-lockup.tsx"
    ).read_text(encoding="utf-8")
    global_css = (WEB_ROOT / "src" / "app" / "globals.css").read_text(
        encoding="utf-8"
    )
    navigation_source = (
        WEB_ROOT / "src" / "components" / "navigation.tsx"
    ).read_text(encoding="utf-8")
    page_source = (WEB_ROOT / "src" / "app" / "page.tsx").read_text(encoding="utf-8")
    journey_source = (
        WEB_ROOT / "src" / "components" / "decision-journey.tsx"
    ).read_text(encoding="utf-8")
    workspace_source = (PROJECT_ROOT / "lab_workspaces.py").read_text(
        encoding="utf-8"
    )
    layout_source = (WEB_ROOT / "src" / "app" / "layout.tsx").read_text(
        encoding="utf-8"
    )
    og_source = (WEB_ROOT / "src" / "app" / "opengraph-image.tsx").read_text(
        encoding="utf-8"
    )

    assert Path(manifest["exports"]["standardSvg"]).name in app_source
    assert Path(manifest["exports"]["favicon128"]).name in app_source
    assert 'aria-hidden="true"' in app_source

    assert "brand-geometry.generated" in component_source
    assert "export function BrandMark" in component_source
    assert "export function BrandLockup" in lockup_source
    assert "BrandMark" in lockup_source
    assert "decorative" in component_source
    assert "aria-hidden" in component_source
    assert "aria-label" in component_source
    for public_type in (
        "BrandMarkProps",
        "BrandMarkVariant",
        "BrandMarkTone",
        "BrandMarkMotion",
    ):
        assert public_type in component_source
    for public_type in ("BrandLockupProps", "BrandLockup"):
        assert public_type in lockup_source
    assert "prefers-reduced-motion" in global_css
    assert "BrandLockup" in navigation_source
    assert "BrandLockup" in page_source
    assert "Know who may leave." not in page_source
    assert "Decide with care." not in page_source
    assert 'className="hero-heading-line"' in page_source
    assert '<span aria-hidden="true">·</span>' not in page_source
    assert ".hero-heading-line" in global_css
    assert ".hero-eyebrow span + span::before" in global_css

    # Editorial display headings use labels, not sentence punctuation. Body copy,
    # policy statements, and the one intentional question heading remain untouched.
    static_next_headings = re.findall(
        r"<h[1-6][^>]*>([^<{]+)</h[1-6]>", page_source
    )
    journey_headings = re.findall(r'title:\s*"([^"]+)"', journey_source)
    static_streamlit_headings = re.findall(
        r"<h[1-6][^>]*>([^<{]+)</h[1-6]>", app_source + workspace_source
    )
    markdown_streamlit_headings = re.findall(
        r'st\.markdown\("#{2,6}\s+([^"]+)"\)', app_source + workspace_source
    )
    display_headings = (
        static_next_headings
        + journey_headings
        + static_streamlit_headings
        + markdown_streamlit_headings
    )
    assert display_headings
    assert not [heading for heading in display_headings if heading.rstrip().endswith(".")]
    assert "What most shaped this fitted model?" in static_next_headings
    assert "The agent can propose — the gate decides what may proceed" in static_next_headings
    assert "brand-geometry.generated" in og_source
    assert "source-serif-4-semibold.ttf" in og_source
    assert "ibm-plex-mono-medium.ttf" in og_source
    assert "Arial" not in og_source

    assert 'from "next/font/local"' in layout_source
    assert 'from "next/font/google"' not in layout_source
    for font_name in (
        "source-serif-4-latin.woff2",
        "source-serif-4-semibold.ttf",
        "ibm-plex-sans-latin.woff2",
        "ibm-plex-mono-500-latin.woff2",
    ):
        assert font_name in layout_source
        assert (WEB_ROOT / "src" / "app" / font_name).is_file()

    for font_name in (
        "source-serif-4-semibold.ttf",
        "ibm-plex-mono-medium.ttf",
    ):
        font_path = WEB_ROOT / "src" / "app" / font_name
        assert font_path.is_file()
        assert font_path.read_bytes()[:4] in {b"\x00\x01\x00\x00", b"OTTO"}


def test_brand_build_is_deterministic_and_part_of_the_next_production_build() -> None:
    package = json.loads((WEB_ROOT / "package.json").read_text(encoding="utf-8"))

    assert package["scripts"]["brand:build"] == "node scripts/build-brand.mjs"
    assert package["scripts"]["brand:check"] == "node scripts/build-brand.mjs --check"
    assert "brand:check" in package["scripts"]["build"]
    assert package["devDependencies"]["sharp"]
    assert (WEB_ROOT / "scripts" / "build-brand.mjs").is_file()


def test_legacy_brand_assets_geometry_and_references_are_gone() -> None:
    remaining_brand_files = {path.name for path in BRAND_DIR.iterdir() if path.is_file()}
    assert LEGACY_BRAND_FILENAMES.isdisjoint(remaining_brand_files)

    production_sources = [
        PROJECT_ROOT / "app.py",
        WEB_ROOT / "src" / "components" / "brand-mark.tsx",
        WEB_ROOT / "src" / "app" / "icon.svg",
        WEB_ROOT / "src" / "app" / "opengraph-image.tsx",
        WEB_ROOT / "src" / "app" / "globals.css",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in production_sources)

    for legacy_name in LEGACY_BRAND_FILENAMES:
        assert legacy_name not in combined

    for legacy_geometry in (
        "M8 27C21 27 28 38 44 43",
        "M8 69C21 69 28 58 44 53",
        "M40.2 28.2A28 28 0 1 1 40.2 67.8",
        "M48 49L56 57L72 38",
        "brand-path-upper",
        "brand-path-lower",
        "brand-core",
        "brand-check",
        "Irish Banking Churn decision flow mark",
        "Atlantic Ledger governed decision-flow mark",
    ):
        assert legacy_geometry not in combined
