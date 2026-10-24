import json
import re
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from ddgs import DDGS
from openai import OpenAI

st.set_page_config(page_title="Homeopathy Medicine Finder", page_icon="🌿", layout="wide")

DISCLAIMER = (
    "The information shown is gathered from publicly available sources and summarized by AI. "
    "Always consult a qualified homeopathic practitioner or licensed medical professional "
    "before starting, stopping, or changing any treatment."
)
UNAVAILABLE = "Information unavailable"
ENGINES = ["Tavily", "Serper", "DuckDuckGo"]
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# ---------- styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Public+Sans:wght@400;500;600&display=swap');
html, body, [class*="st-"]:not([data-testid="stIconMaterial"]) { font-family: 'Public Sans', sans-serif; }
/* keep Streamlit's icon font intact, otherwise ligature names like "arrow_right" show as text */
[data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    font-feature-settings: 'liga'; letter-spacing: normal; text-transform: none; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; color: #2F5446; }
.hero { padding: 1.4rem 1.6rem; border-radius: 18px; margin-bottom: 1rem;
        background: linear-gradient(120deg, #DCE8DF 0%, #E6EDF3 100%); }
.hero h1 { margin: 0; font-size: 2rem; }
.hero p { margin: .3rem 0 0; color: #51635A; }
.stButton > button, .stDownloadButton > button { background: #3F6B5C; color: #fff; border: 0;
        border-radius: 12px; padding: .55rem 1rem; font-weight: 600; transition: background .15s; }
.stButton > button:hover, .stDownloadButton > button:hover { background: #2F5446; color: #fff; }
.card { background: #fff; border: 1px solid #DCE5DD; border-radius: 14px; padding: 1rem 1.2rem; }
.card .lbl { color: #6B7B72; font-size: .85rem; margin-bottom: .2rem; }
.card .val { font-size: 1.05rem; }
.badge { display: inline-block; padding: .15rem .65rem; border-radius: 999px; font-size: .82rem; font-weight: 600; }
.b-na { background: #F3E8D0; color: #7A5C12; }
.b-high { background: #D5E8DB; color: #1F5A3A; }
.b-mod { background: #DCE6F1; color: #2E557E; }
.b-low { background: #F3E8D0; color: #7A5C12; }
.b-none { background: #EDE3E3; color: #7A3434; }
.foot { margin-top: 2rem; padding: 1rem 1.2rem; border-radius: 12px; background: #E8EEE7;
        color: #44554C; font-size: .88rem; text-align: center; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- helpers ----------
def secret(key: str) -> str:
    try:
        return str(st.secrets.get(key, "") or "")
    except Exception:
        return ""


def domain(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def badge(text: str, cls: str) -> str:
    return f'<span class="badge {cls}">{text}</span>'


def na_badge() -> str:
    return badge(f"⚠ {UNAVAILABLE}", "b-na")


# ---------- search layer ----------
def _tavily(query: str) -> list[dict]:
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": secret("TAVILY_API_KEY"), "query": query,
              "search_depth": "advanced", "max_results": 8},
        timeout=40,
    )
    r.raise_for_status()
    return [{"url": x["url"], "title": x.get("title", ""), "text": x.get("content", "")}
            for x in r.json().get("results", [])]


def _page_text(url: str) -> str:
    try:
        html = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}).text
        html = re.sub(r"(?is)<(script|style|nav|footer|header).*?</\1>", " ", html)
        return re.sub(r"\s+", " ", re.sub(r"(?s)<[^>]+>", " ", html))[:4000]
    except Exception:
        return ""


def _serper(query: str, fetch_pages: bool) -> list[dict]:
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": secret("SERPER_API_KEY"), "Content-Type": "application/json"},
        json={"q": query, "gl": "in", "num": 8},
        timeout=30,
    )
    r.raise_for_status()
    items = [{"url": x["link"], "title": x.get("title", ""), "text": x.get("snippet", "")}
             for x in r.json().get("organic", [])]
    return _enrich(items) if fetch_pages else items


def _enrich(items: list[dict]) -> list[dict]:
    """Serper/DuckDuckGo return only snippets, so fetch page bodies for a deeper search."""
    with ThreadPoolExecutor(6) as ex:
        for it, body in zip(items, ex.map(lambda i: _page_text(i["url"]), items)):
            if len(body) > len(it["text"]):
                it["text"] = body
    return items


def _ddg(query: str, fetch_pages: bool) -> list[dict]:
    hits = DDGS().text(query, region="in-en", max_results=8)
    items = [{"url": x["href"], "title": x.get("title", ""), "text": x.get("body", "")} for x in hits]
    return _enrich(items) if fetch_pages else items


@st.cache_data(ttl=86400, show_spinner=False)
def web_search(engine: str, query: str, deep: bool = False) -> list[dict]:
    if engine == "Tavily":
        return _tavily(query)
    if engine == "Serper":
        return _serper(query, deep)
    return _ddg(query, deep)


# ---------- LLM layer ----------
def llm(system: str, user: str) -> str:
    client = OpenAI(
        api_key=secret("OPENROUTER_API_KEY"), base_url=OPENROUTER_URL,
        default_headers={"HTTP-Referer": "http://localhost:8501", "X-Title": "Homeopathy Medicine Finder"},
    )
    resp = client.chat.completions.create(
        model=secret("OPENROUTER_MODEL"), temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"[\[{].*[\]}]", raw, re.S)
    return json.loads(m.group(0) if m else raw)


@st.cache_data(ttl=86400, show_spinner=False)
def find_candidates(engine: str, query: str) -> list[str]:
    results = web_search(engine, f"{query} homeopathic medicine remedy name")
    ctx = "\n".join(f"- {r['title']}: {r['text'][:300]}" for r in results)
    raw = llm(
        "You are an expert in homeopathy in India. From the search snippets ONLY, list up to 8 distinct "
        "homeopathic medicine names that match the user's partial or full name. Use only names that "
        "literally appear in the snippets. Return a JSON array of strings, nothing else. "
        "Return [] if nothing fits.",
        f"User input: {query}\n\nSnippets:\n{ctx}",
    )
    seen, out = set(), []
    for n in parse_json(raw):
        if isinstance(n, str) and n.strip() and n.strip().lower() not in seen:
            seen.add(n.strip().lower())
            out.append(n.strip())
    return out


FIELDS = ["name", "usage", "dosage", "duration", "side_effects", "interactions_homeopathy", "interactions_allopathy"]


@st.cache_data(ttl=86400, show_spinner=False)
def build_report(engine: str, medicine: str) -> dict:
    queries = [
        f"{medicine} homeopathic medicine uses indications materia medica India",
        f"{medicine} homeopathy dosage 30CH how to take duration",
        f"{medicine} homeopathy side effects aggravation antidote incompatible remedies allopathic interaction",
    ]
    with ThreadPoolExecutor(3) as ex:
        batches = list(ex.map(lambda q: web_search(engine, q, True), queries))
    sources, seen = [], set()
    for b in batches:
        for r in b:
            if r["url"] not in seen and r["text"].strip():
                seen.add(r["url"])
                sources.append(r)
    sources = sources[:14]
    ctx = "\n\n".join(f"[{i+1}] {s['url']}\n{s['text'][:3000]}" for i, s in enumerate(sources))
    schema = {k: {"text": "string or null", "sources": [1]} for k in FIELDS}
    raw = llm(
        "You are an expert in homeopathy in India. Using ONLY the numbered sources provided, write a report "
        "on the medicine. Rules: (1) never use outside knowledge; (2) if a field is not clearly supported by the "
        "sources set its text to null; (3) 'sources' lists the numbers of the sources supporting that field; "
        "(4) the default potency is 30CH - report dosage for 30CH when stated, and if sources only mention other "
        "potencies, say so explicitly; (5) keep text concise, use short markdown bullets where helpful; "
        f"(6) interactions_homeopathy = clashes/incompatibilities/antidotes with other homeopathic remedies; "
        "interactions_allopathy = clashes with allopathic drugs; (7) never claim 'no side effects' or 'no "
        f"interactions' unless a source explicitly says so. Return ONLY JSON in this shape: {json.dumps(schema)}",
        f"Medicine: {medicine}\n\nSources:\n{ctx}",
    )
    data = parse_json(raw)
    rep = {"medicine": medicine, "fields": {}, "sources": sources}
    cited = set()
    for k in FIELDS:
        f = data.get(k) or {}
        text = (f.get("text") or "").strip() if isinstance(f, dict) else ""
        ids = [i for i in (f.get("sources") or []) if isinstance(i, int) and 1 <= i <= len(sources)] if text else []
        if text and not ids:
            text = ""  # unsupported claim: drop rather than show
        rep["fields"][k] = {"text": text, "sources": ids}
        cited.update(ids)
    rep["cited"] = sorted(cited)
    return rep


# ---------- report rendering ----------
def confidence(rep: dict) -> tuple[str, str, int]:
    n = len({domain(rep["sources"][i - 1]["url"]) for i in rep["cited"]})
    if n >= 4:
        return "High confidence", "b-high", n
    if n >= 2:
        return "Moderate confidence", "b-mod", n
    if n == 1:
        return "Low confidence", "b-low", n
    return "No usable evidence", "b-none", n


def report_text(rep: dict) -> str:
    f = rep["fields"]
    label, _, n = confidence(rep)
    def v(k): return f[k]["text"] or UNAVAILABLE
    parts = [
        f"Medicine: {f['name']['text'] or rep['medicine']}",
        f"Default potency considered: 30CH",
        f"Confidence: {label} ({n} independent sources)",
        f"\nUSAGE\n{v('usage')}", f"\nDOSAGE (30CH default)\n{v('dosage')}",
        f"\nHOW LONG TO USE AT A STRETCH\n{v('duration')}", f"\nSIDE EFFECTS\n{v('side_effects')}",
        f"\nCLASH WITH OTHER HOMEOPATHIC MEDICINES\n{v('interactions_homeopathy')}",
        f"\nCLASH WITH ALLOPATHIC MEDICINES\n{v('interactions_allopathy')}",
        "\nSOURCES", *[f"- {rep['sources'][i-1]['url']}" for i in rep["cited"]],
        f"\nDISCLAIMER: {DISCLAIMER}",
    ]
    return "\n".join(parts)


def make_pdf(rep: dict) -> bytes:
    def clean(s: str) -> str:
        s = s.replace("•", "-").replace("**", "").replace("`", "")
        return s.encode("latin-1", "replace").decode("latin-1")
    pdf = FPDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, clean(f"Homeopathy Report: {rep['medicine']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for line in clean(report_text(rep)).split("\n"):
        if line.isupper() and line.strip():
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
        else:
            pdf.multi_cell(0, 5.5, line or " ", new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def copy_button(text: str):
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        """<button id="b" style="background:#3F6B5C;color:#fff;border:0;border-radius:12px;padding:.6rem 1rem;
        font:600 15px sans-serif;width:100%;cursor:pointer">📋 Copy Report</button>
        <script>const t=__T__;const b=document.getElementById('b');
        b.onclick=async()=>{try{await navigator.clipboard.writeText(t)}catch(e){const a=document.createElement('textarea');
        a.value=t;document.body.appendChild(a);a.select();document.execCommand('copy');a.remove()}
        b.innerText='Copied ✓';setTimeout(()=>b.innerText='📋 Copy Report',1800)}</script>""".replace("__T__", payload),
        height=50,
    )


def src_links(rep: dict, ids: list[int]) -> str:
    return " ".join(f"[{i}]({rep['sources'][i-1]['url']})" for i in ids)


def section(rep: dict, title: str, key: str, expanded=False):
    fld = rep["fields"][key]
    with st.expander(title, expanded=expanded):
        if fld["text"]:
            st.markdown(fld["text"])
            st.caption(f"Sources: {src_links(rep, fld['sources'])}")
        else:
            st.markdown(na_badge(), unsafe_allow_html=True)


def render_report(rep: dict):
    f = rep["fields"]
    label, cls, n = confidence(rep)
    st.subheader(f["name"]["text"] or rep["medicine"])
    st.markdown(
        badge(label, cls) + " &nbsp; " + badge(f"{n} independent source{'s' if n != 1 else ''}", "b-mod")
        + " &nbsp; " + badge(f"{len(rep['sources'])} pages reviewed", "b-mod"),
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2)
    c1.markdown('<div class="card"><div class="lbl">Default potency</div><div class="val">30CH</div></div>',
                unsafe_allow_html=True)
    dur = f["duration"]["text"]
    c2.markdown(
        f'<div class="card"><div class="lbl">How long to use at a stretch</div>'
        f'<div class="val">{dur if dur else na_badge()}</div></div>', unsafe_allow_html=True)
    st.write("")
    section(rep, "🌱 Usage", "usage", expanded=True)
    section(rep, "💧 Dosage (30CH default)", "dosage", expanded=True)
    section(rep, "⚠️ Side Effects", "side_effects")
    with st.expander("🔀 Interactions & Clashes"):
        for title, key in [("With other homeopathic medicines", "interactions_homeopathy"),
                           ("With allopathic medicines", "interactions_allopathy")]:
            st.markdown(f"**{title}**")
            fld = f[key]
            if fld["text"]:
                st.markdown(fld["text"])
                st.caption(f"Sources: {src_links(rep, fld['sources'])}")
            else:
                st.markdown(na_badge(), unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns(3)
    with a:
        st.download_button("📄 Export PDF", make_pdf(rep), f"{rep['medicine']}_report.pdf",
                           "application/pdf", width="stretch")
    with b:
        copy_button(report_text(rep))
    with c:
        if st.button("🔄 Refresh search", width="stretch"):
            build_report.clear()
            st.rerun()
    with st.expander("Plain text (use the copy icon if the Copy button is blocked)"):
        st.code(report_text(rep), language=None, wrap_lines=True)
    st.markdown("### 🔗 Sources")
    st.caption(f"{len(rep['cited'])} cited of {len(rep['sources'])} pages reviewed. Verify the details yourself:")
    for i in rep["cited"]:
        s = rep["sources"][i - 1]
        st.markdown(f"{i}. [{s['title'] or domain(s['url'])}]({s['url']}) — `{domain(s['url'])}`")
    uncited = [s for i, s in enumerate(rep["sources"], 1) if i not in rep["cited"]]
    if uncited:
        with st.expander(f"Other pages reviewed but not cited ({len(uncited)})"):
            for s in uncited:
                st.markdown(f"- [{s['title'] or domain(s['url'])}]({s['url']})")


# ---------- state ----------
ss = st.session_state
ss.setdefault("candidates", None)
ss.setdefault("selected", None)
ss.setdefault("history", [])
ss.setdefault("engine", "Tavily")
ss.setdefault("query", "")


def pick(name: str):
    ss.selected = name
    ss.history = [h for h in ss.history if h["name"] != name]
    ss.history.insert(0, {"name": name, "engine": ss.engine})
    ss.history = ss.history[:15]


def load_history(h: dict):
    ss.engine, ss.selected, ss.candidates = h["engine"], h["name"], None


# ---------- sidebar ----------
with st.sidebar:
    st.markdown("### 🕘 Search history")
    if not ss.history:
        st.caption("Your searches will appear here.")
    for i, h in enumerate(ss.history):
        st.button(h["name"], key=f"h{i}", on_click=load_history, args=(h,), width="stretch")
    if ss.history and st.button("Clear history", width="stretch"):
        ss.history = []
        st.rerun()

# ---------- main ----------
st.markdown(
    '<div class="hero"><h1>🌿 Homeopathy Medicine Finder</h1>'
    "<p>Type a full or partial medicine name. Every detail is traced back to a web source.</p></div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([4, 2, 1.3], vertical_alignment="bottom")
ss.query = c1.text_input("Medicine name", value=ss.query, placeholder="e.g. Arnica, Nux, Bryo…")
ss.engine = c2.radio("Search engine", ENGINES, horizontal=True, index=ENGINES.index(ss.engine))
go = c3.button("🔍 Search", width="stretch")

needed = ["OPENROUTER_MODEL", "OPENROUTER_API_KEY"]
if ss.engine == "Tavily":
    needed.append("TAVILY_API_KEY")
elif ss.engine == "Serper":
    needed.append("SERPER_API_KEY")  # DuckDuckGo needs no key
missing = [k for k in needed if not secret(k)]
if missing:
    st.error(f"Missing in .streamlit/secrets.toml: {', '.join(missing)}")
    st.stop()

if go and ss.query.strip():
    ss.selected = None
    try:
        with st.spinner("Searching the web for matching medicines…"):
            ss.candidates = find_candidates(ss.engine, ss.query.strip())
    except Exception as e:
        ss.candidates = None
        st.error(f"Search failed: {e}")

if ss.candidates is not None and not ss.selected:
    if ss.candidates:
        st.markdown("#### Did you mean…")
    else:
        st.markdown(f"{na_badge()} No matching medicine names were found online.", unsafe_allow_html=True)
    options = ss.candidates + ([ss.query.strip()] if ss.query.strip() and ss.query.strip() not in ss.candidates else [])
    cols = st.columns(3)
    for i, name in enumerate(options):
        label = name if name in ss.candidates else f"Use “{name}” as typed"
        cols[i % 3].button(label, key=f"c{i}", on_click=pick, args=(name,), width="stretch")

if ss.selected:
    try:
        with st.spinner(f"Running a deep search on {ss.selected}…"):
            rep = build_report(ss.engine, ss.selected)
        render_report(rep)
    except Exception as e:
        st.error(f"Could not build the report: {e}")

st.markdown(f'<div class="foot">⚕️ {DISCLAIMER}</div>', unsafe_allow_html=True)