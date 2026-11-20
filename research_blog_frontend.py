from __future__ import annotations

import json
import re
import time
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

from research_blog_backend import app  # your compiled LangGraph app

# ============================================================
# Page config + theme
# ============================================================
st.set_page_config(
    page_title="Blog Writing Agent",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
:root{
  --accent:#7c5cff; --accent2:#22d3ee; --ok:#22c55e; --warn:#f59e0b;
  --card:rgba(127,127,127,.08); --line:rgba(127,127,127,.22); --muted:rgba(127,127,127,.95);
}
.block-container{padding-top:1.6rem; max-width:1250px;}
#MainMenu, footer{visibility:hidden;}

/* hero */
.hero{
  border-radius:22px; padding:30px 34px; margin-bottom:18px; color:#fff;
  background:linear-gradient(120deg,#5b3df5 0%,#7c5cff 45%,#22d3ee 130%);
  box-shadow:0 14px 40px rgba(91,61,245,.28);
}
.hero h1{margin:0; font-size:2.1rem; font-weight:800; letter-spacing:-.02em; color:#fff;}
.hero p{margin:.45rem 0 0; font-size:1.02rem; opacity:.92;}
.hero .tags span{
  display:inline-block; margin:12px 8px 0 0; padding:3px 12px; border-radius:99px;
  background:rgba(255,255,255,.18); font-size:.78rem; font-weight:600; backdrop-filter:blur(4px);
}

/* pipeline stepper */
.stepper{display:flex; gap:10px; flex-wrap:wrap; margin:6px 0 14px;}
.step{
  flex:1; min-width:130px; padding:12px 14px; border-radius:14px;
  border:1px solid var(--line); background:var(--card); transition:all .25s;
}
.step .ic{font-size:1.25rem}
.step .nm{font-weight:700; font-size:.9rem; margin-top:2px}
.step .st{font-size:.72rem; color:var(--muted); text-transform:uppercase; letter-spacing:.06em}
.step.done{border-color:rgba(34,197,94,.55); background:rgba(34,197,94,.09)}
.step.done .st{color:var(--ok)}
.step.active{border-color:var(--accent); background:rgba(124,92,255,.14); animation:pulse 1.3s infinite}
.step.active .st{color:var(--accent)}
.step.skip{opacity:.45}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(124,92,255,.45)}50%{box-shadow:0 0 0 7px rgba(124,92,255,0)}}

/* metric cards */
.metrics{display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:8px 0 18px}
.metric{border:1px solid var(--line); background:var(--card); border-radius:16px; padding:14px 16px}
.metric .v{font-size:1.7rem; font-weight:800; line-height:1.1}
.metric .l{font-size:.78rem; color:var(--muted); margin-top:3px; text-transform:uppercase; letter-spacing:.05em}

/* cards */
.card{border:1px solid var(--line); background:var(--card); border-radius:16px; padding:16px 18px; margin-bottom:12px}
.card h4{margin:0 0 4px; font-size:1.02rem}
.card .goal{color:var(--muted); font-size:.92rem; margin-bottom:8px}
.pill{
  display:inline-block; padding:2px 10px; margin:0 6px 6px 0; border-radius:99px; font-size:.72rem;
  font-weight:600; border:1px solid var(--line); background:rgba(127,127,127,.1);
}
.pill.on{background:rgba(124,92,255,.16); border-color:rgba(124,92,255,.5); color:var(--accent)}
.ev a{text-decoration:none; font-weight:700}
.ev .meta{font-size:.78rem; color:var(--muted)}
.ev .snip{font-size:.88rem; margin-top:6px}

/* article */
.article{max-width:820px; margin:0 auto; padding:6px 4px}
.article h1{font-size:2.2rem; letter-spacing:-.02em}
.article img{border-radius:14px}

section[data-testid="stSidebar"] .stButton>button{border-radius:12px}
.stButton>button[kind="primary"]{
  background:linear-gradient(120deg,#5b3df5,#7c5cff); border:none; font-weight:700; padding:.6rem 1rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Streamlit API compat (width="stretch" replaced use_container_width) ---
try:
    _v = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _v = (1, 0)
STRETCH: Dict[str, Any] = {"width": "stretch"} if _v >= (1, 50) else {"use_container_width": True}

IMAGES_DIR = Path("images")

EXAMPLES = [
    "How KV caching speeds up LLM inference",
    "Latest open-source LLM releases this week",
    "RAG vs fine-tuning: a practical comparison",
    "Designing a multi-agent system with LangGraph",
]

STAGES = [
    ("router", "🧭", "Router"),
    ("research", "🔎", "Research"),
    ("orchestrator", "🧩", "Planner"),
    ("worker", "✍️", "Writers"),
    ("reducer", "🖼️", "Editor"),
]

# ============================================================
# Helpers
# ============================================================
def safe_slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9 _-]+", "", title.strip().lower())
    return re.sub(r"\s+", "_", s).strip("_") or "blog"


def word_count(md: str) -> int:
    return len(re.findall(r"\w+", re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)))


def zip_bytes(files: Dict[str, bytes], folder: Optional[Path] = None) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        if folder and folder.is_dir():
            for p in folder.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def to_dict(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file() and p.name.lower() != "readme.md"]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)

def is_valid_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False


def show_image(path, caption=None):
    p = Path(path)

    if p.exists() and is_valid_image(p):
        st.image(str(p), caption=caption, **STRETCH)
    else:
        st.warning(
            f"{p.name} is missing or not a valid image. "
            "Delete it and regenerate."
        )


# ---------- markdown with local images ----------
_IMG = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*(?:\n\*(?P<cap>[^*\n]+)\*)?")


def render_markdown(md: str) -> None:
    pos = 0
    for m in _IMG.finditer(md):
        before = md[pos : m.start()]
        if before.strip():
            st.markdown(before)
        src, alt, cap = m.group("src").strip(), m.group("alt").strip(), (m.group("cap") or "").strip()
        if src.startswith(("http://", "https://")):
            st.image(src, caption=cap or alt or None, **STRETCH)
        else:
            path = Path(src.lstrip("./")).resolve()
            show_image(path, cap or alt or None)
        pos = m.end()
    tail = md[pos:]
    if tail.strip():
        st.markdown(tail)


# ---------- pipeline stepper ----------
def stepper_html(done: set, skipped: set, active: Optional[str], extra: Dict[str, str]) -> str:
    out = ['<div class="stepper">']
    for key, icon, name in STAGES:
        if key in skipped:
            cls, label = "skip", "skipped"
        elif key in done:
            cls, label = "done", extra.get(key, "done")
        elif key == active:
            cls, label = "active", extra.get(key, "running…")
        else:
            cls, label = "", "waiting"
        out.append(
            f'<div class="step {cls}"><div class="ic">{icon}</div>'
            f'<div class="nm">{name}</div><div class="st">{label}</div></div>'
        )
    out.append("</div>")
    return "".join(out)


def metrics_html(items: List[Tuple[str, str]]) -> str:
    cells = "".join(f'<div class="metric"><div class="v">{v}</div><div class="l">{l}</div></div>' for l, v in items)
    return f'<div class="metrics">{cells}</div>'


# ---------- single-pass graph streaming ----------
def run_graph(inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    """
    Streams the graph ONCE (no second invoke -> no double LLM cost).
    Yields ("node", node_name, update) and ("state", full_state).
    """
    state: Dict[str, Any] = dict(inputs)
    for mode, payload in app.stream(inputs, stream_mode=["updates", "values"]):
        if mode == "updates":
            for node, upd in payload.items():
                yield ("node", node, upd)
        else:
            state = payload
            yield ("state", state, None)
    yield ("final", state, None)


# ============================================================
# Session state
# ============================================================
st.session_state.setdefault("last_out", None)
st.session_state.setdefault("logs", [])
st.session_state.setdefault("topic", "")
st.session_state.setdefault("run_meta", {})


def _set_topic(t: str) -> None:
    st.session_state["topic"] = t


def _load_blog(p: Path) -> None:
    md = p.read_text(encoding="utf-8", errors="replace")
    st.session_state["last_out"] = {"plan": None, "evidence": [], "image_specs": [], "final": md}
    st.session_state["run_meta"] = {"loaded": p.name}


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### ✍️ New blog")
    st.text_area(
        "Topic",
        key="topic",
        height=120,
        placeholder="e.g. How does speculative decoding work? (add a word limit like “500 words” if you want)",
    )

    st.caption("Quick ideas")
    for ex in EXAMPLES:
        st.button(ex, key=f"ex_{ex}", on_click=_set_topic, args=(ex,), **STRETCH)

    with st.expander("⚙️ Research settings"):
        as_of = st.date_input("As-of date", value=date.today())
        recency_days = st.slider("Recency window (days)", 1, 60, 7, help="Used for open-book / news roundups.")

    run_btn = st.button("🚀 Generate blog", type="primary", **STRETCH)

    st.divider()
    st.markdown("### 📚 Past blogs")
    past = list_past_blogs()
    if not past:
        st.caption("No saved blogs yet (*.md in this folder).")
    else:
        q = st.text_input("Search", placeholder="Filter by title…", label_visibility="collapsed")
        shown = 0
        for p in past[:50]:
            try:
                t = title_from_md(p.read_text(encoding="utf-8", errors="replace"), p.stem)
            except Exception:
                t = p.stem
            if q and q.lower() not in t.lower():
                continue
            st.button(f"📄 {t[:48]}", key=f"load_{p.name}", on_click=_load_blog, args=(p,), **STRETCH)
            shown += 1
            if shown >= 12:
                break

# ============================================================
# Hero
# ============================================================
st.markdown(
    """
<div class="hero">
  <h1>Blog Writing Agent</h1>
  <p>Route → research → plan → write in parallel → illustrate. Powered by LangGraph.</p>
  <div class="tags"><span>LangGraph</span><span>Groq</span><span>Tavily</span><span>Parallel workers</span><span>Auto images</span></div>
</div>
""",
    unsafe_allow_html=True,
)

# ============================================================
# Run
# ============================================================
if run_btn:
    topic = st.session_state["topic"].strip()
    if not topic:
        st.warning("Please enter a topic first.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic,
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": int(recency_days),
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    st.session_state["logs"] = []
    live = st.container(border=True)
    with live:
        st.markdown(f"**Generating:** {topic}")
        step_ph = st.empty()
        info_ph = st.empty()
        bar = st.progress(0.0)

    done: set = set()
    skipped: set = set()
    extra: Dict[str, str] = {}
    active: Optional[str] = "router"
    t0 = time.time()
    final_state: Dict[str, Any] = {}
    error: Optional[Exception] = None

    def refresh(state: Dict[str, Any]) -> None:
        plan = to_dict(state.get("plan"))
        total = len(plan["tasks"]) if plan else 0
        got = len(state.get("sections") or [])
        if total:
            extra["worker"] = f"{got}/{total} sections"
            bar.progress(min(0.35 + 0.55 * got / total, 0.9))
        if state.get("mode"):
            extra["router"] = state["mode"].replace("_", " ")
        if "research" in done:
            extra["research"] = f"{len(state.get('evidence') or [])} sources"
        if plan:
            extra["orchestrator"] = f"{total} tasks"
        step_ph.markdown(stepper_html(done, skipped, active, extra), unsafe_allow_html=True)

    refresh(inputs)

    try:
        for kind, a, b in run_graph(inputs):
            if kind == "node":
                node = a
                st.session_state["logs"].append(f"[{time.time() - t0:5.1f}s] node={node} {json.dumps(b, default=str)[:600]}")
                if node == "router":
                    done.add("router")
                    if b and not b.get("needs_research"):
                        skipped.add("research")
                        active = "orchestrator"
                        bar.progress(0.2)
                    else:
                        active = "research"
                        bar.progress(0.12)
                elif node == "research":
                    done.add("research"); active = "orchestrator"; bar.progress(0.25)
                elif node == "orchestrator":
                    done.add("orchestrator"); active = "worker"; bar.progress(0.35)
                elif node == "worker":
                    active = "worker"
                elif node == "reducer":
                    done.add("worker"); done.add("reducer"); active = None; bar.progress(1.0)
            elif kind == "state":
                refresh(a)
                done_workers = a.get("plan") and len(a.get("sections") or []) >= len(to_dict(a["plan"])["tasks"])
                if done_workers and "worker" not in done:
                    done.add("worker"); active = "reducer"
                info_ph.caption(f"⏱️ {time.time() - t0:0.1f}s elapsed")
            elif kind == "final":
                final_state = a
    except Exception as e:  # noqa: BLE001
        error = e

    if error:
        bar.empty()
        st.error(f"Run failed: {error}")
        with st.expander("Details"):
            st.exception(error)
    else:
        done.update({"router", "orchestrator", "worker", "reducer"} | ({"research"} if "research" not in skipped else set()))
        active = None
        refresh(final_state)
        bar.progress(1.0)
        info_ph.success(f"Done in {time.time() - t0:0.1f}s")
        st.session_state["last_out"] = final_state
        st.session_state["run_meta"] = {"seconds": round(time.time() - t0, 1)}

# ============================================================
# Results
# ============================================================
out = st.session_state.get("last_out")

if not out:
    st.markdown(
        """
<div class="card" style="text-align:center; padding:40px">
  <div style="font-size:2.4rem">📝</div>
  <h4>Nothing generated yet</h4>
  <div class="goal">Enter a topic in the sidebar (or pick a quick idea) and hit <b>Generate blog</b>.</div>
</div>""",
        unsafe_allow_html=True,
    )
    st.stop()

plan = to_dict(out.get("plan"))
evidence = [to_dict(e) for e in (out.get("evidence") or [])]
specs = out.get("image_specs") or []
final_md: str = out.get("final") or ""
blog_title = (plan or {}).get("blog_title") or title_from_md(final_md, "blog")
n_images = len(list(IMAGES_DIR.glob("*"))) if IMAGES_DIR.exists() else 0
wc = word_count(final_md)

st.markdown(
    metrics_html(
        [
            ("Words", f"{wc:,}"),
            ("Read time", f"{max(1, round(wc / 220))} min"),
            ("Sections", str(len(plan["tasks"])) if plan else "—"),
            ("Sources", str(len(evidence))),
            ("Images", str(len(specs) or n_images)),
            ("Mode", (out.get("mode") or "—").replace("_", " ")),
        ]
    ),
    unsafe_allow_html=True,
)

tab_preview, tab_plan, tab_evidence, tab_images, tab_logs = st.tabs(
    ["📝 Blog", "🧩 Plan", "🔎 Evidence", "🖼️ Images", "🧾 Logs"]
)

# ---------- Blog ----------
with tab_preview:
    if not final_md:
        st.warning("No final markdown found.")
    else:
        c1, c2, c3 = st.columns([1, 1, 4])
        md_name = f"{safe_slug(blog_title)}.md"
        c1.download_button("⬇️ Markdown", final_md.encode("utf-8"), md_name, "text/markdown", **STRETCH)
        c2.download_button(
            "📦 Bundle",
            zip_bytes({md_name: final_md.encode("utf-8")}, IMAGES_DIR),
            f"{safe_slug(blog_title)}_bundle.zip",
            "application/zip",
            **STRETCH,
        )
        with c3:
            with st.popover("📋 Copy raw markdown"):
                st.code(final_md, language="markdown")

        st.markdown('<div class="article">', unsafe_allow_html=True)
        render_markdown(final_md)
        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Plan ----------
with tab_plan:
    if not plan:
        st.info("No plan available (blogs loaded from disk don't store their plan).")
    else:
        a, b, c = st.columns(3)
        a.markdown(f'<div class="card"><div class="goal">Audience</div><h4>{plan.get("audience")}</h4></div>', unsafe_allow_html=True)
        b.markdown(f'<div class="card"><div class="goal">Tone</div><h4>{plan.get("tone")}</h4></div>', unsafe_allow_html=True)
        c.markdown(f'<div class="card"><div class="goal">Blog kind</div><h4>{plan.get("blog_kind")}</h4></div>', unsafe_allow_html=True)

        if plan.get("constraints"):
            st.caption("Constraints: " + " · ".join(plan["constraints"]))

        tasks = sorted(plan.get("tasks", []), key=lambda t: t.get("id", 0))
        total_words = sum(t.get("target_words", 0) for t in tasks)
        st.caption(f"{len(tasks)} sections · ~{total_words} target words")

        for t in tasks:
            pills = "".join(
                f'<span class="pill on">{lbl}</span>'
                for lbl, on in [
                    ("research", t.get("requires_research")),
                    ("citations", t.get("requires_citations")),
                    ("code", t.get("requires_code")),
                ]
                if on
            ) + "".join(f'<span class="pill">{tag}</span>' for tag in (t.get("tags") or []))
            bullets = "".join(f"<li>{x}</li>" for x in t.get("bullets", []))
            st.markdown(
                f'<div class="card"><h4>{t.get("id")}. {t.get("title")} '
                f'<span class="pill">{t.get("target_words")} words</span></h4>'
                f'<div class="goal">{t.get("goal")}</div>{pills}<ul style="margin:6px 0 0 1rem">{bullets}</ul></div>',
                unsafe_allow_html=True,
            )

        df = pd.DataFrame(tasks)[["id", "title", "target_words"]] if tasks else None
        if df is not None:
            st.bar_chart(df.set_index("title")["target_words"], horizontal=True)

# ---------- Evidence ----------
with tab_evidence:
    if not evidence:
        st.info("No evidence (closed-book mode, no Tavily key, or nothing within the recency window).")
    else:
        for e in evidence:
            meta = " · ".join(x for x in [e.get("source"), e.get("published_at")] if x) or "date unknown"
            snip = f'<div class="snip">{e["snippet"]}</div>' if e.get("snippet") else ""
            st.markdown(
                f'<div class="card ev"><a href="{e.get("url")}" target="_blank">{e.get("title") or e.get("url")}</a>'
                f'<div class="meta">{meta}</div>{snip}</div>',
                unsafe_allow_html=True,
            )
        st.download_button(
            "⬇️ Evidence (CSV)",
            pd.DataFrame(evidence).to_csv(index=False).encode("utf-8"),
            "evidence.csv",
            "text/csv",
        )

# ---------- Images ----------
with tab_images:
    files = sorted(p for p in IMAGES_DIR.iterdir() if p.is_file()) if IMAGES_DIR.exists() else []
    if not specs and not files:
        st.info("No images for this blog.")
    else:
        if files:
            cols = st.columns(2)
            for i, p in enumerate(files):
                with cols[i % 2]:
                    show_image(p, p.name)
            st.download_button(
                "⬇️ Images (zip)", zip_bytes({}, IMAGES_DIR), "images.zip", "application/zip"
            )
        if specs:
            with st.expander("Image plan / prompts"):
                st.json(specs)

# ---------- Logs ----------
with tab_logs:
    logs = st.session_state.get("logs", [])
    if not logs:
        st.caption("Run a generation to see node-by-node logs.")
    else:
        st.code("\n\n".join(logs[-80:]), language="text")