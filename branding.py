"""Branding for Pragya - the ONE place for name, tagline, logo and colours.

Change anything about how the app presents itself here, and nowhere else.
Rules from the brief: no government emblems, and never the words
"Official" or "CBDT" in the branding.
"""
from pathlib import Path

APP_NAME = "Pragya"
APP_NAME_HINDI = "प्रज्ञा"
APP_MEANING = "insight"
TAGLINE = "Insight for every tax desk"

# One accent colour (teal) and one font everywhere.
ACCENT = "#0F766E"
ACCENT_DARK_MODE = "#2DD4BF"   # same teal family, readable on a dark background
ACCENT_SOFT = "#F0FDFA"        # light tint for chips / highlights
FONT_FAMILY = "Inter"

# Risk colours used on every screen: red / amber / grey = high / medium / low.
RISK_COLOURS = {"HIGH": "#DC2626", "MEDIUM": "#D97706", "LOW": "#A8A29E"}

# Every generated draft carries this mark (the officer decides).
DRAFT_MARK = "DRAFT - for the officer's review and decision"

# Shown wherever demo data is in use.
DEMO_DATA_LABEL = "Synthetic demo data - not real taxpayers or offices"

# Words that must never appear in the branding (checked by tests).
FORBIDDEN_BRANDING_WORDS = ("official", "cbdt")

# Original, simple logo: a teal rounded square with a four-point "insight" spark.
# Plain SVG so it works offline. No emblems or seals.

def _icon_svg(size: int = 40) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 40 40">'
            f'<rect x="4" y="4" width="32" height="32" rx="9" fill="{ACCENT}"/>'
            '<path d="M20 10 L22.6 17.4 L30 20 L22.6 22.6 L20 30 L17.4 22.6 L10 20 L17.4 17.4 Z" fill="#FFFFFF"/>'
            '<circle cx="27.5" cy="12.5" r="2.2" fill="#FFFFFF" opacity=".75"/></svg>')


def _logo_svg(text_colour: str = ACCENT) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="170" height="40" viewBox="0 0 170 40">'
            f'<rect x="2" y="4" width="32" height="32" rx="9" fill="{ACCENT}"/>'
            '<path d="M18 10 L20.6 17.4 L28 20 L20.6 22.6 L18 30 L15.4 22.6 L8 20 L15.4 17.4 Z" fill="#FFFFFF"/>'
            '<circle cx="25.5" cy="12.5" r="2.2" fill="#FFFFFF" opacity=".75"/>'
            f'<text x="44" y="27" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="20" '
            f'font-weight="700" fill="{text_colour}">{APP_NAME}</text></svg>')


def asset_paths(folder: Path) -> dict:
    """Write the logo files (from the SVG above) into `folder` and return their paths.
    The app calls this at start-up, so the files always match this module."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {"icon": (folder / "pragya_icon.svg", _icon_svg()),
             "logo": (folder / "pragya_logo.svg", _logo_svg()),
             "logo_dark": (folder / "pragya_logo_dark.svg", _logo_svg(ACCENT_DARK_MODE))}
    out = {}
    for key, (path, svg) in files.items():
        if not path.exists() or path.read_text(encoding="utf-8") != svg:
            path.write_text(svg, encoding="utf-8")
        out[key] = str(path)
    return out


def page_title() -> str:
    """Browser-tab title, e.g. 'Pragya - Insight for every tax desk'."""
    return f"{APP_NAME} - {TAGLINE}"
