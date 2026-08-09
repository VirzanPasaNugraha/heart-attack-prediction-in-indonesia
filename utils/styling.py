"""
Modul styling & komponen visual untuk CardioSense AI.

Identitas visual:
- Palet: navy-teal klinis (primary) + coral-merah "denyut nadi" (accent),
  di atas latar putih-abu bersih. Bukan cream/terracotta generik, bukan
  dark-mode neon generik - dipilih supaya terasa medis, presisi, terpercaya.
- Tipografi: Fraunces (display, serif berkarakter, dipakai terbatas di judul)
  dipasangkan dengan IBM Plex Sans (body) dan IBM Plex Mono (angka/metrik) -
  kombinasi yang umum dipakai produk health-tech/data untuk kesan presisi.
- Elemen tanda tangan (signature element): garis gelombang EKG/heartbeat
  yang muncul berulang sebagai divider - elemen ini konsisten mengikat
  seluruh halaman ke tema jantung.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Garis EKG (signature element) - dipakai sebagai divider & aksen di banyak
# tempat. Digambar manual sebagai gelombang detak jantung yang berulang.
# ---------------------------------------------------------------------------
_EKG_BEAT = "L{x0} 50 L{x1} 50 L{x2} 20 L{x3} 80 L{x4} 12 L{x5} 60 L{x6} 50 L{x7} 50 "


def _ekg_path(n_beats: int = 6, beat_width: int = 200) -> str:
    d = "M0 50 "
    for i in range(n_beats):
        base = i * beat_width
        d += _EKG_BEAT.format(
            x0=base + 55,
            x1=base + 70,
            x2=base + 80,
            x3=base + 90,
            x4=base + 100,
            x5=base + 110,
            x6=base + 120,
            x7=base + beat_width,
        )
    return d


def ekg_divider(color: str = "var(--cs-accent)", height: int = 28, opacity: float = 1.0) -> str:
    """Kembalikan HTML garis pemisah bermotif EKG selebar penuh."""
    width = 1200
    d = _ekg_path(n_beats=6, beat_width=width // 6)
    return f"""
    <div style="width:100%; line-height:0; opacity:{opacity};">
        <svg viewBox="0 0 {width} 100" preserveAspectRatio="none"
             style="width:100%; height:{height}px; display:block;">
            <path d="{d}" fill="none" stroke="{color}" stroke-width="4"
                  stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
    </div>
    """


def inject_base_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
            --cs-bg: #F6F8FA;
            --cs-surface: #FFFFFF;
            --cs-surface-alt: #EEF3F3;
            --cs-ink: #14232F;
            --cs-ink-soft: #4B5D6B;
            --cs-primary: #0E4749;
            --cs-primary-light: #1B6B73;
            --cs-accent: #E0483F;
            --cs-accent-soft: #FBE4E2;
            --cs-risk-low: #1F7A54;
            --cs-risk-low-soft: #E3F3EB;
            --cs-risk-medium: #C98A1D;
            --cs-risk-medium-soft: #FBF0DC;
            --cs-risk-high: #C23B32;
            --cs-risk-high-soft: #FBE2E0;
            --cs-border: #DCE3E8;
            --cs-font-display: 'Fraunces', Georgia, serif;
            --cs-font-body: 'IBM Plex Sans', -apple-system, sans-serif;
            --cs-font-mono: 'IBM Plex Mono', 'Courier New', monospace;
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            font-family: var(--cs-font-body);
            color: var(--cs-ink);
        }

        [data-testid="stAppViewContainer"] { background: var(--cs-bg); }

        h1, h2, h3 {
            font-family: var(--cs-font-display) !important;
            color: var(--cs-primary) !important;
            font-weight: 600 !important;
            letter-spacing: -0.01em;
        }
        h4, h5, h6 { font-family: var(--cs-font-body); color: var(--cs-ink); }

        [data-testid="stSidebar"] {
            background: var(--cs-primary);
            border-right: 1px solid var(--cs-border);
        }
        [data-testid="stSidebar"] * { color: #EAF3F2 !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #C9DEDC !important; }

        /* Metrik */
        [data-testid="stMetric"] {
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 14px;
            padding: 14px 16px 10px 16px;
        }
        [data-testid="stMetricValue"] {
            font-family: var(--cs-font-mono) !important;
            color: var(--cs-primary) !important;
        }
        [data-testid="stMetricLabel"] { color: var(--cs-ink-soft) !important; }

        /* Tombol */
        div.stButton > button, [data-testid="stFormSubmitButton"] button, [data-testid="stBaseButton-secondaryFormSubmit"] {
            background: var(--cs-accent) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6em 1.4em !important;
            transition: transform 0.12s ease, box-shadow 0.12s ease;
            box-shadow: 0 2px 0 rgba(194,59,50,0.35);
        }
        div.stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(224,72,63,0.35);
        }

        /* Expander & container */
        [data-testid="stExpander"] {
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 12px;
        }

        /* Divider tipis */
        hr { border-color: var(--cs-border) !important; }

        /* Scroll-margin biar rapi */
        section.main > div { padding-top: 1.2rem; }

        @media (max-width: 640px) {
            h1 { font-size: 1.6rem !important; }
            h2 { font-size: 1.25rem !important; }
            .cs-hero-title { font-size: 1.6rem !important; }
            .cs-hero-subtitle { font-size: 0.92rem !important; }
            .cs-hero-banner { padding: 1.6rem 1.3rem 1.2rem 1.3rem !important; }
            .cs-section-title { font-size: 1.25rem !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_banner(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="cs-hero-banner" style="
            background: linear-gradient(135deg, var(--cs-primary) 0%, #0A3335 100%);
            border-radius: 20px;
            padding: 2.4rem 2.2rem 1.6rem 2.2rem;
            margin-bottom: 1.6rem;
            position: relative;
            overflow: hidden;
        ">
            <div style="font-family:var(--cs-font-mono); letter-spacing:0.14em; text-transform:uppercase;
                        font-size:0.75rem; color:#8FCFC7; margin-bottom:0.5rem;">{eyebrow}</div>
            <div class="cs-hero-title" style="font-family:var(--cs-font-display); font-weight:700; font-size:2.4rem;
                        color:#FFFFFF; line-height:1.15; margin-bottom:0.6rem;">{title}</div>
            <div class="cs-hero-subtitle" style="font-family:var(--cs-font-body); font-size:1.02rem; color:#C9DEDC; max-width:640px;">{subtitle}</div>
            <div style="margin-top:1.4rem;">{ekg_divider(color="#E0483F", height=26, opacity=0.85)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(eyebrow: str, title: str):
    st.markdown(
        f"""
        <div style="margin: 0.4rem 0 0.9rem 0;">
            <div style="font-family:var(--cs-font-mono); letter-spacing:0.12em; text-transform:uppercase;
                        font-size:0.72rem; color:var(--cs-accent); margin-bottom:0.15rem;">{eyebrow}</div>
            <div class="cs-section-title" style="font-family:var(--cs-font-display); font-weight:600; font-size:1.55rem; color:var(--cs-primary);">{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(icon: str, title: str, body: str, accent: str = "primary"):
    color_var = {"primary": "var(--cs-primary)", "accent": "var(--cs-accent)"}[accent]
    st.markdown(
        f"""
        <div style="
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-left: 4px solid {color_var};
            border-radius: 12px;
            padding: 1.1rem 1.2rem;
            height: 100%;
        ">
            <div style="font-size:1.5rem; margin-bottom:0.35rem;">{icon}</div>
            <div style="font-weight:600; color:var(--cs-ink); margin-bottom:0.3rem;">{title}</div>
            <div style="color:var(--cs-ink-soft); font-size:0.92rem; line-height:1.5;">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    """level: 'low' | 'medium' | 'high' -> potongan HTML pill berwarna."""
    palette = {
        "low": ("var(--cs-risk-low)", "var(--cs-risk-low-soft)", "Risiko Rendah"),
        "medium": ("var(--cs-risk-medium)", "var(--cs-risk-medium-soft)", "Risiko Sedang"),
        "high": ("var(--cs-risk-high)", "var(--cs-risk-high-soft)", "Risiko Tinggi"),
    }
    color, soft, label = palette[level]
    return (
        f'<span style="background:{soft}; color:{color}; font-weight:700; '
        f'padding:0.35em 0.9em; border-radius:999px; font-size:0.95rem; '
        f'font-family:var(--cs-font-mono);">{label}</span>'
    )


def dev_card(name: str, role: str, blurb: str, links: list[tuple[str, str]] | None = None):
    links_html = ""
    if links:
        chips = "".join(
            f'<a href="{url}" target="_blank" style="text-decoration:none;">'
            f'<span style="display:inline-block; margin:4px 6px 0 0; padding:0.3em 0.8em; '
            f'border:1px solid var(--cs-border); border-radius:999px; font-size:0.82rem; '
            f'color:var(--cs-primary); background:var(--cs-surface-alt);">{label}</span></a>'
            for label, url in links
        )
        links_html = f'<div style="margin-top:0.7rem;">{chips}</div>'

    st.markdown(
        f"""
        <div style="
            background: var(--cs-surface);
            border: 1px solid var(--cs-border);
            border-radius: 18px;
            padding: 1.8rem;
            display:flex; gap:1.4rem; align-items:center; flex-wrap:wrap;
        ">
            <div style="
                width:76px; height:76px; border-radius:50%;
                background: linear-gradient(135deg, var(--cs-primary), var(--cs-primary-light));
                display:flex; align-items:center; justify-content:center;
                font-family:var(--cs-font-display); font-size:1.8rem; color:#fff; font-weight:700;
                flex-shrink:0;
            ">{"".join([w[0] for w in name.split()[:2]])}</div>
            <div style="flex:1; min-width:200px;">
                <div style="font-family:var(--cs-font-display); font-size:1.35rem; font-weight:700; color:var(--cs-ink);">{name}</div>
                <div style="font-family:var(--cs-font-mono); font-size:0.85rem; color:var(--cs-accent); margin-bottom:0.5rem;">{role}</div>
                <div style="color:var(--cs-ink-soft); font-size:0.94rem; line-height:1.55;">{blurb}</div>{links_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
