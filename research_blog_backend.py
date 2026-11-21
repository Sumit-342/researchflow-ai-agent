from __future__ import annotations

import operator
import os
import re
import unicodedata
import hashlib
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated , Dict , Tuple,Any

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


from typing import List, Optional
from datetime import date, timedelta
from PIL import Image
import json
from pydantic import ValidationError

from langchain_community.tools import TavilySearchResults

import os
import base64
import requests
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should be able to do/understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete, non-overlapping subpoints to cover in this section.",
    )
    target_words: int = Field(..., description="Target word count for this section (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # keep if Tavily provides; DO NOT rely on it
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)



class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Short descriptive name, e.g. kv_cache_flow")
    kind: Literal["diagram", "illustration"] = "diagram"
    alt: str
    caption: str
    after_heading: str = Field(
        ..., description="Exact Markdown section heading after which the image should appear."
    )
    diagram_code: str = Field(
        "", description="Mermaid source (no code fences). Required when kind='diagram'."
    )
    prompt: str = Field(
        "", description="Text-free visual prompt. Required when kind='illustration'."
    )



class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)



class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str


# -----------------------------
# 2) LLM
# -----------------------------
llm = ChatGroq( model="openai/gpt-oss-120b")
small_llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

def structured_invoke(schema, messages, retries=3,model = None):
    """
    Robust structured output for Groq.
    Tries JSON Schema first, then function calling.
    """

    last_err = None

    for method in ("json_schema", "function_calling"):
        try:
            selected_llm = model or llm

            chain = selected_llm.with_structured_output(
                schema,
                method=method
            )

            for _ in range(retries):
                try:
                    return chain.invoke(messages)

                except Exception as e:
                    last_err = e

                    salvaged = _salvage_failed_generation(
                        e, schema
                    )

                    if salvaged is not None:
                        return salvaged

        except Exception as e:
            last_err = e

    raise last_err


def _salvage_failed_generation(err, schema):
    """
    Attempts to recover JSON from Groq's failed_generation.
    """

    try:
        body = getattr(err, "body", None) or {}

        raw = (
            (body.get("error") or body)
            .get("failed_generation")
        )

        if not raw:
            return None

        data = json.loads(raw)

        if isinstance(data, dict) and "arguments" in data:
            data = data["arguments"]

        return schema.model_validate(data)

    except (ValueError, ValidationError, AttributeError):
        return None

# -----------------------------
# 3) Router 
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:

    topic = state["topic"]

    decision = structured_invoke(
        RouterDecision,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {topic}"),
        ],
        model=small_llm,
    )

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------

def _tavily_search(
    query: str,
    max_results: int = 2,
) -> List[dict]:

    if not os.getenv("TAVILY_API_KEY"):
        return []

    try:
        tool = TavilySearchResults(
            max_results=max_results
        )

        results = tool.invoke(
            {"query": query}
        )

        normalized = []

        for r in results or []:

            snippet = (
                r.get("content")
                or r.get("snippet")
                or ""
            )

            normalized.append(
                {
                    "title": (
                        r.get("title") or ""
                    )[:200],

                    "url": r.get("url") or "",

                    "snippet": snippet[:300],

                    "published_at": (
                        r.get("published_date")
                        or r.get("published_at")
                    ),

                    "source": r.get("source"),
                }
            )

        return normalized

    except Exception as e:
        print(f"Tavily search failed: {e}")
        return []


def _iso_to_date(
    s: Optional[str]
) -> Optional[date]:

    if not s:
        return None

    try:
        return date.fromisoformat(s[:10])

    except (ValueError, TypeError):
        return None


def research_node(state: State) -> dict:

    # Limit queries to control search volume
    queries = (
        state.get("queries") or []
    )[:3]

    raw_results = []

    # Search with Tavily
    for q in queries:

        raw_results.extend(
            _tavily_search(
                q,
                max_results=2
            )
        )

    if not raw_results:
        return {"evidence": []}

    # Deduplicate by URL
    seen_urls = set()
    evidence = []

    for r in raw_results:

        url = (
            r.get("url") or ""
        ).strip()

        if not url or url in seen_urls:
            continue

        seen_urls.add(url)

        # Normalize publication date
        published_date = _iso_to_date(
            r.get("published_at")
        )

        published_at = (
            published_date.isoformat()
            if published_date
            else None
        )

        evidence.append(
            EvidenceItem(
                title=r.get("title", ""),
                url=url,
                snippet=(
                    r.get("snippet") or ""
                )[:300],
                published_at=published_at,
                source=r.get("source"),
            )
        )

    # Maximum evidence limit
    evidence = evidence[:6]

    # Preserve original open_book recency filtering
    if state.get("mode") == "open_book":

        as_of = date.fromisoformat(
            state["as_of"]
        )

        cutoff = as_of - timedelta(
            days=int(state["recency_days"])
        )

        evidence = [
            e
            for e in evidence
            if (
                _iso_to_date(e.published_at)
                and _iso_to_date(e.published_at) >= cutoff
            )
        ]

    return {
        "evidence": evidence
    }

# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 3-5 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (100–250)

If the topic contains a word limit such as
"180 words" or "500 words",

the TOTAL blog length must stay within
that budget.

Divide the budget across all sections.

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:

    evidence = state.get("evidence", [])
    mode = state.get("mode", "closed_book")

    plan = structured_invoke(
        Plan,
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n\n"
                    f"Evidence (ONLY use for fresh claims; may be empty):\n"
                    f"{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ],
    )

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker (write one section)
# -----------------------------
WORKER_SYSTEM = """
You are a senior technical writer.

Write ONE markdown section.

Rules:
- Start with '## Section Title'
- Cover all bullets in order.
- Stay near target_words.
- Output only markdown.

If blog_kind='news_roundup':
- Summarize events and implications.
- Do not write a tutorial.

If mode='open_book':
- Use only provided evidence URLs for claims.
- Cite with: [Source](URL).

If requires_code=true:
- Include a minimal correct code example.

Keep writing concise, technical, and practical.

Citation Rules:
- Every cited claim must use a URL from Evidence.
- Never create empty citations.
- Never invent URLs.
- If evidence is insufficient, write:
  "Not found in provided sources."
- The section heading must exactly match the provided Section title.

"""


def worker_node(payload: dict) -> dict:
    
    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic = payload["topic"]
    mode = payload.get("mode", "closed_book")

    bullets_text = "\n- " + "\n- ".join(task.bullets)

    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = small_llm.invoke(
        [
            SystemMessage(content=WORKER_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog title: {plan.blog_title}\n"
                    f"Audience: {plan.audience}\n"
                    f"Tone: {plan.tone}\n"
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Constraints: {plan.constraints}\n"
                    f"Topic: {topic}\n"
                    f"Mode: {mode}\n\n"
                    f"Section title: {task.title}\n"
                    f"Goal: {task.goal}\n"
                    f"Target words: {task.target_words}\n"
                    f"Tags: {task.tags}\n"
                    f"requires_research: {task.requires_research}\n"
                    f"requires_citations: {task.requires_citations}\n"
                    f"requires_code: {task.requires_code}\n"
                    f"Bullets:{bullets_text}\n\n"
                    f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
                )
            ),
        ]
    ).content.strip()

    return {"sections": [(task.id, section_md)]}


# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:

    plan = state["plan"]

    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.

Decide whether images would materially improve the reader's understanding of THIS blog.

Rules:
- Generate 1 to 2 images. Prefer 1. Never create decorative filler.
- Do not rewrite or return the blog markdown.
- Set after_heading to an EXACT heading copied from the blog.
- Use kind="diagram" for architectures, flows, pipelines, comparisons, timelines.
  Provide diagram_code as Mermaid and leave prompt empty.
- Use kind="illustration" ONLY for a conceptual hero visual.
  Provide prompt and leave diagram_code empty.

Mermaid rules (diagram_code):
- Start with `flowchart LR` or `flowchart TD`. No code fences.
- At most 8 nodes. Labels under 5 words.
- Always quote labels: A["Query vectors"] --> B["Attention scores"]
- Edge labels must be quoted too: A -->|"softmax"| B
- No parentheses, colons, or special characters outside quotes.

Illustration prompt rules:
- Describe shapes, objects, colors, composition and style only.
- NEVER request text, words, letters, numbers, labels or logos in the image.

Return strictly GlobalImagePlan.
"""
# ============================================================
# Helper Function
# ============================================================
def _norm_heading(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\u2010-\u2015\u2212]", "-", s)
    s = re.sub(r"^#{1,6}\s*", "", s.strip())
    return re.sub(r"\s+", " ", s).casefold()


def insert_image_placeholders(
    md: str,
    image_specs: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:

    lines = md.splitlines()
    insertions = {}
    valid_specs = []
    seen_placeholders = set()

    for spec in image_specs:
        heading = spec.get("after_heading", "").strip()
        placeholder = spec.get("placeholder", "").strip()

        # Validate placeholder format
        if not re.fullmatch(r"\[\[IMAGE_\d+\]\]", placeholder):
            continue

        # Prevent duplicate placeholders
        if placeholder in seen_placeholders or placeholder in md:
            continue

        
        # Find heading using normalized text
        target = _norm_heading(heading)

        matches = [
            i for i, line in enumerate(lines)
            if re.match(r"^#{1,6}\s+", line.strip())
            and _norm_heading(line) == target
        ]

        # Skip missing or ambiguous headings
        if len(matches) != 1:
            continue

        heading_index = matches[0]

        insertions.setdefault(heading_index + 1, []).append(placeholder)

        seen_placeholders.add(placeholder)
        valid_specs.append(spec)

    # Build Markdown without rewriting its original content
    result = []

    for i, line in enumerate(lines):
        result.append(line)

        if i + 1 in insertions:
            result.append("")
            result.extend(insertions[i + 1])
            result.append("")

    return "\n".join(result) + "\n", valid_specs



def decide_images(state: State) -> dict:

    plan = state["plan"]
    assert plan is not None

    merged_md = state["merged_md"]

    image_plan = structured_invoke(
        GlobalImagePlan,
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Here is the complete blog markdown:\n\n"
                    f"{merged_md}\n\n"
                    "Decide whether  1, or 2 images would improve "
                    "this blog.\n\n"
                    "For every image:\n"
                    "- Provide a unique placeholder: [[IMAGE_1]], "
                    "[[IMAGE_2]].\n"
                    "- Provide a descriptive filename.\n"
                    "- Write a useful alt text and caption.\n"
                    "- Write a detailed image-generation prompt.\n"
                    "- Set after_heading to an EXACT heading copied "
                    "from the blog markdown.\n\n"
                    "Do not rewrite, summarize, or return the blog."
                )
            ),
        ],
        model = small_llm
    )

    image_specs = [
        img.model_dump()
        for img in image_plan.images
    ]

    md_with_placeholders, valid_specs = insert_image_placeholders(
        state["merged_md"],
        image_specs
    )

    return {
        "md_with_placeholders": md_with_placeholders,
        "image_specs": valid_specs
    }

def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    NVIDIA FLUX.3 Klein image generation.
    Returns validated image bytes.
    """

    api_key = os.getenv("NVIDIA_API_KEY")

    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set.")

    prompt = (
    f"{prompt.strip()}. Clean flat vector-style illustration, "
    "no text, no letters, no words, no labels, no watermark."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

    url = (
        "https://ai.api.nvidia.com/v1/genai/"
        "black-forest-labs/flux-3-image-klein"
    )

    prompt = (
    f"{prompt.strip()}. "
    "Clean technical illustration, professional visual composition, "
    "no text, no letters, no labels, no numbers, no watermark."
    )

    payload = {
        "prompt": prompt,
        "width": 1024,
        "height": 1024,
        "steps": 20,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    artifacts = data.get("artifacts", [])

    if not artifacts:
        raise RuntimeError(
            f"NVIDIA returned no image artifacts. "
            f"Response keys: {list(data.keys())}"
        )

    artifact = artifacts[0]
    image_b64 = artifact.get("base64")

    if not image_b64:
        raise RuntimeError("NVIDIA returned empty image data.")

    try:
        image_bytes = base64.b64decode(
            image_b64,
            validate=True
        )
    except Exception as e:
        raise RuntimeError(
            f"Invalid Base64 image data: {e}"
        ) from e

    if _sniff_ext(image_bytes) is None:
        raise RuntimeError(
            "Decoded data is not a supported image. "
            f"First 16 bytes: {image_bytes[:16].hex()}"
        )

    return image_bytes



def safe_image_filename(filename: str) -> str:
    stem = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        Path(Path(filename).name).stem
    ).strip("_")

    return stem or "image"

def _sniff_ext(b: bytes) -> Optional[str]:
    if b.startswith(b"\xff\xd8\xff"):
        return ".jpg"

    if b.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png"

    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return ".webp"

    return None


def _is_valid_image(p: Path) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:
        return False



def _mermaid_png_bytes(code: str) -> bytes:
    code = re.sub(r"^```(?:mermaid)?\s*|\s*```$", "", code.strip())
    if not code:
        raise RuntimeError("Empty Mermaid diagram.")

    token = base64.urlsafe_b64encode(code.encode("utf-8")).decode("ascii")
    resp = requests.get(
        f"https://mermaid.ink/img/{token}",
        params={"type": "png", "bgColor": "FFFFFF", "width": 1200},
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(
            f"mermaid.ink returned {resp.status_code} (check Mermaid syntax)."
        )

    data = resp.content
    if _sniff_ext(data) != ".png":
        raise RuntimeError("mermaid.ink did not return a PNG.")
    return data


def _render_spec(spec: dict) -> bytes:
    if spec.get("kind") == "diagram" and spec.get("diagram_code"):
        return _mermaid_png_bytes(spec["diagram_code"])

    return _gemini_generate_image_bytes(
        spec.get("prompt") or spec.get("caption", "")
    )


def _spec_stem(spec: dict) -> str:
    # Hash in the name so a stale image from another blog is never reused by accident.
    base = safe_image_filename(spec["filename"])
    src = spec.get("diagram_code") or spec.get("prompt") or ""
    return f"{base}_{hashlib.sha1(src.encode('utf-8')).hexdigest()[:8]}"



def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None

    # Use Python-prepared Markdown
    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # Create images directory only when needed
    images_dir = Path("images")

    if image_specs:
        images_dir.mkdir(exist_ok=True)

    for spec in image_specs:
        placeholder = spec["placeholder"]
        stem = _spec_stem(spec)

        # Reuse an existing valid image
        existing = next(
            (
                p for p in images_dir.glob(f"{stem}.*")
                if _is_valid_image(p)
            ),
            None
        )

        if existing:
            filename = existing.name

        else:
            try:
                img_bytes = _render_spec(spec)

                ext = _sniff_ext(img_bytes)

                if ext is None:
                    raise RuntimeError("Unsupported image format.")

                filename = stem + ext

                (images_dir / filename).write_bytes(img_bytes)

            except Exception as e:
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** "
                    f"{spec.get('caption', '')}\n>\n"
                    f"> **Alt:** {spec.get('alt', '')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt', '')}\n>\n"
                    f"> **Error:** {e}\n"
                )

                md = md.replace(placeholder, prompt_block)
                continue

        md = md.replace(
            placeholder,
            f"![{spec['alt']}](images/{filename})\n"
            f"*{spec['caption']}*"
        )

    # Safe blog filename
    blog_filename = (
        re.sub(r"[^a-z0-9_-]+", "_", plan.blog_title.lower())
        .strip("_")
        or "blog"
    ) + ".md"

    Path(blog_filename).write_text(
        md,
        encoding="utf-8"
    )

    return {"final": md}



# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app