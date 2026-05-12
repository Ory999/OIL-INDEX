"""
Script 20 — LLM Multidimensional Scoring
Model: Locally hosted openai/gpt-oss-20b via ngrok tunnel

Resumable — if interrupted, re-running skips already scored documents.
Historical backfill loaded from data/Historic/ — only new documents scored.
Incremental saves every 50 documents — max 50 documents lost on crash.

ALIGNMENT NOTE:
  SYSTEM_PROMPT and build_user_prompt() are locked to match the historic
  backfill exactly (prompt_version hash: 0d23c19bbf76d4ee62cc685480ea43f0).

  reasoning_effort="high" and RESPONSE_SCHEMA are passed explicitly in the
  API call because these were active during the historic backfill via the
  LM Studio Semesterprojekt preset (Custom Fields > Reasoning Effort = High,
  Structured Output = enabled with the schema below). Passing them in the
  API call guarantees identical inference behaviour regardless of which
  preset is active in LM Studio at run time.

  Do NOT modify SYSTEM_PROMPT, RESPONSE_SCHEMA, reasoning_effort, or
  build_user_prompt() without re-scoring the full historic corpus.
"""
import os, json, logging, time, re, hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import httpx
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR      = Path(os.getenv("DATA_DIR",     "data/raw"))
HISTORIC_DIR = Path(os.getenv("HISTORIC_DIR", "data/Historic"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "openai/gpt-oss-20b")
MAX_RETRIES  = 3
RETRY_DELAY  = 5.0
SAVE_EVERY   = 50

SOURCE_CONTEXT = {
    "OPEC_MOMR": "OPEC Monthly Oil Market Report — official production cartel assessment",
    "ARAMCO":    "Saudi Aramco press coverage — world's largest oil producer",
    "EIA_STEO":  "EIA Short-Term Energy Outlook — US government official oil market forecast",
}

# ── JSON RESPONSE SCHEMA ──────────────────────────────────────────────────────
# Matches the Structured Output schema in the LM Studio Semesterprojekt preset.
# The preset enforced this schema during the historic backfill at inference time.
# Passed explicitly in the API call so GitHub Actions gets identical behaviour
# regardless of which preset is loaded in LM Studio at run time.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "oil_impact_score":              {"type": "number", "minimum": -1, "maximum": 1},
        "supply_disruption_signal":      {"type": "number", "minimum": -1, "maximum": 1},
        "demand_outlook_signal":         {"type": "number", "minimum": -1, "maximum": 1},
        "geopolitical_risk_signal":      {"type": "number", "minimum": -1, "maximum": 1},
        "surface_vs_implied_divergence": {"type": "number", "minimum": 0,  "maximum": 1},
        "institutional_confidence":      {"type": "number", "minimum": 0,  "maximum": 1},
        "dominant_theme": {
            "type": "string",
            "enum": ["SUPPLY_CONCERN", "DEMAND_WEAKNESS", "GEOPOLITICAL",
                     "PRODUCTION_CUT", "PRODUCTION_INCREASE", "MARKET_BALANCE",
                     "PRICE_FORECAST", "SANCTIONS", "NEUTRAL"]
        },
        "reasoning": {"type": "string"}
    },
    "required": [
        "oil_impact_score", "supply_disruption_signal", "demand_outlook_signal",
        "geopolitical_risk_signal", "surface_vs_implied_divergence",
        "institutional_confidence", "dominant_theme", "reasoning"
    ]
}

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────
# LOCKED to match historic backfill. Hash: 0d23c19bbf76d4ee62cc685480ea43f0
# Do NOT add, remove, or reword any line without re-scoring the historic corpus.
SYSTEM_PROMPT = """You are a quantitative analyst specialising in WTI crude oil commodity markets with 20 years of experience at a major energy trading desk.

Your task is to analyse official institutional communications and extract structured sentiment signals that quantify their directional impact on WTI crude oil prices.

SCORING FRAMEWORK — ALL SCORES ARE ON A STRICT -1.0 TO +1.0 SCALE:
- -1.0 = Maximum bearish signal (extreme downward pressure on oil prices)
- -0.5 = Moderately bearish
-  0.0 = Neutral / no directional signal
- +0.5 = Moderately bullish
- +1.0 = Maximum bullish signal (extreme upward pressure on oil prices)

SCORE DEFINITIONS:
1. oil_impact_score (-1 to +1): production cuts (+), supply disruptions (+), demand growth (+) vs production increases (-), demand weakness (-), oversupply (-)
2. supply_disruption_signal (-1 to +1): +1 = severe supply disruption, -1 = supply surplus, 0 = balanced
3. demand_outlook_signal (-1 to +1): +1 = strong demand growth, -1 = demand destruction, 0 = stable
4. geopolitical_risk_signal (-1 to +1): +1 = high geopolitical risk, -1 = de-escalation, 0 = no developments
5. surface_vs_implied_divergence (0 to 1): GAP between surface language and implied market reality
6. institutional_confidence (0 to 1): 1.0 = definitive statements, 0.5 = hedged, 0.0 = vague

CRITICAL RULES:
— Return ONLY valid JSON, no other text, no markdown
— All scores except surface_vs_implied_divergence must be between -1.0 and +1.0
— surface_vs_implied_divergence must be between 0.0 and 1.0"""

# Verify prompt has not drifted from the historic backfill version
PROMPT_VERSION          = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()
PROMPT_VERSION_EXPECTED = "0d23c19bbf76d4ee62cc685480ea43f0"
if PROMPT_VERSION != PROMPT_VERSION_EXPECTED:
    raise RuntimeError(
        f"SYSTEM_PROMPT hash mismatch — live prompt ({PROMPT_VERSION}) does not match "
        f"historic backfill ({PROMPT_VERSION_EXPECTED}). "
        f"Re-score the historic corpus before deploying this change."
    )


def build_user_prompt(text: str, source: str, date: str) -> str:
    """
    LOCKED to match historic backfill format.
    Simple document + JSON template — no analytical guidance questions.
    """
    source_context = SOURCE_CONTEXT.get(source, f"Official institutional statement — {source}")
    text_excerpt   = text[:2000].strip()
    return f"""DOCUMENT ANALYSIS REQUEST

Source type: {source_context}
Publication date: {date}
Text excerpt:
\"\"\"
{text_excerpt}
\"\"\"

Return ONLY this JSON object with no other text:
{{
    "oil_impact_score":              <float -1.0 to 1.0>,
    "supply_disruption_signal":      <float -1.0 to 1.0>,
    "demand_outlook_signal":         <float -1.0 to 1.0>,
    "geopolitical_risk_signal":      <float -1.0 to 1.0>,
    "surface_vs_implied_divergence": <float 0.0 to 1.0>,
    "institutional_confidence":      <float 0.0 to 1.0>,
    "dominant_theme":                <string: SUPPLY_CONCERN|DEMAND_WEAKNESS|GEOPOLITICAL|PRODUCTION_CUT|PRODUCTION_INCREASE|MARKET_BALANCE|PRICE_FORECAST|SANCTIONS|NEUTRAL>,
    "reasoning":                     <string: max 50 words>
}}"""


def neutral_scores(reason: str = "LLM failed after retries") -> dict:
    return {
        "oil_impact_score":              0.0,
        "supply_disruption_signal":      0.0,
        "demand_outlook_signal":         0.0,
        "geopolitical_risk_signal":      0.0,
        "surface_vs_implied_divergence": 0.0,
        "institutional_confidence":      0.0,
        "dominant_theme":                "NEUTRAL",
        "reasoning":                     reason,
        "llm_scored":                    False,
        "llm_failed":                    True,
        "prompt_version":                PROMPT_VERSION,
    }


def get_llm_client():
    try:
        client = OpenAI(
            base_url=LLM_BASE_URL,
            api_key="none",
            http_client=httpx.Client(
                headers={"ngrok-skip-browser-warning": "true"},
                timeout=60.0,
            )
        )
        models = client.models.list()
        model  = models.data[0].id if models.data else LLM_MODEL
        log.info(f"✓ Connected to local model: {model} at {LLM_BASE_URL}")
        log.info(f"  Prompt version: {PROMPT_VERSION} ✓ (matches historic backfill)")
        return client, model
    except Exception as e:
        raise ConnectionError(f"Cannot connect to local model at {LLM_BASE_URL}: {e}")


def score_document_with_retry(client, model: str, text: str,
                               source: str, date: str) -> dict:
    """
    API call replicates the LM Studio Semesterprojekt preset settings exactly:
    - temperature=0.1              (Settings > Temperature)
    - max_tokens=400               (Settings > Limit Response Length — off,
                                    but 400 is a safe ceiling for the JSON schema)
    - reasoning_effort="high"      (Custom Fields > Reasoning Effort = High)
    - response_format JSON schema  (Structured Output — enabled with RESPONSE_SCHEMA)
    """
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(text, source, date)},
                ],
                temperature=0.1,
                max_tokens=400,
                reasoning_effort="high",
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name":   "oil_sentiment_scores",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    }
                },
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                raw = re.sub(r'```json?|```', '', raw).strip()
            scores = json.loads(raw)
            scores["llm_scored"]     = True
            scores["llm_failed"]     = False
            scores["prompt_version"] = PROMPT_VERSION
            return scores
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            log.debug(f"  Attempt {attempt}/{MAX_RETRIES} — {last_error}")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            last_error = str(e)
            log.debug(f"  Attempt {attempt}/{MAX_RETRIES} — {e}")
            time.sleep(RETRY_DELAY)

    log.warning(f"  All {MAX_RETRIES} attempts failed — neutral scores ({last_error[:60]})")
    return neutral_scores(f"Failed after {MAX_RETRIES} retries: {last_error[:80]}")


def validate_and_clip_scores(scores: dict) -> dict:
    for field in ["oil_impact_score", "supply_disruption_signal",
                  "demand_outlook_signal", "geopolitical_risk_signal"]:
        if field in scores:
            scores[field] = float(np.clip(scores[field], -1.0, 1.0))
    if "surface_vs_implied_divergence" in scores:
        scores["surface_vs_implied_divergence"] = float(
            np.clip(scores["surface_vs_implied_divergence"], 0.0, 1.0))
    if "institutional_confidence" in scores:
        scores["institutional_confidence"] = float(
            np.clip(scores["institutional_confidence"], 0.0, 1.0))
    return scores


def load_historic_scores() -> pd.DataFrame | None:
    historic_path = HISTORIC_DIR / "llm_scores.parquet"
    if not historic_path.exists():
        log.info("  No historic scores found — scoring all documents fresh")
        return None
    try:
        df = pd.read_parquet(historic_path)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
        log.info(f"✓ Loaded {len(df)} historic scores from {historic_path}")
        return df
    except Exception as e:
        log.warning(f"  Could not load historic scores: {e}")
        return None


def save_incremental(records: list, out: Path, already_scored_df: pd.DataFrame | None):
    if not records:
        return
    new_df = pd.DataFrame(records).set_index("doc_index")
    if already_scored_df is not None and len(already_scored_df) > 0:
        combined = pd.concat([already_scored_df, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.to_parquet(out)
    else:
        new_df.to_parquet(out)


def run_llm_scoring():
    corpus_path = RAW_DIR / "combined_corpus.parquet"
    out         = RAW_DIR / "llm_scores.parquet"

    if not corpus_path.exists():
        log.warning("combined_corpus.parquet not found — run 18 build corpus.py first")
        return pd.DataFrame()

    corpus = pd.read_parquet(corpus_path)
        if len(corpus) == 0:
            log.warning("combined_corpus.parquet is empty — run 18 build corpus.py first")
            return pd.DataFrame()

    corpus["date"] = pd.to_datetime(corpus["date"]).dt.tz_localize(None).dt.normalize()

    # ── Load historic pre-scored documents (date+source matching) ─────────
    historic_df   = load_historic_scores()
    historic_keys = set()
    if historic_df is not None and len(historic_df) > 0:
        historic_keys = set(
            zip(historic_df["date"].dt.date.astype(str),
                historic_df["source"])
        )
        log.info(f"  Historic keys available: {len(historic_keys)} (date+source pairs)")

    # ── Resume from existing daily scores (index-based) ───────────────────
    already_scored_df  = None
    already_scored_idx = set()
    if out.exists():
        try:
            already_scored_df  = pd.read_parquet(out)
            already_scored_idx = set(already_scored_df.index.tolist())
            log.info(f"  Daily resume: {len(already_scored_idx)} already scored today")
        except Exception:
            log.info("  No valid daily scores — starting fresh")

    # ── Determine which documents need LLM scoring ────────────────────────
    def needs_scoring(idx, row) -> bool:
        if idx in already_scored_idx:
            return False
        doc_key = (str(row["date"].date()), str(row.get("source", "")))
        if doc_key in historic_keys:
            return False
        return True

    to_score = [(idx, row) for idx, row in corpus.iterrows()
                if needs_scoring(idx, row)]

    historic_hits = len(corpus) - len(already_scored_idx) - len(to_score)
    log.info(f"\n  Corpus total:          {len(corpus)}")
    log.info(f"  From historic data:    {historic_hits}")
    log.info(f"  Already scored today:  {len(already_scored_idx)}")
    log.info(f"  Need LLM scoring:      {len(to_score)}")

    if len(to_score) == 0:
        log.info("✓ All documents covered by historic or daily scores — nothing to do")
        _save_full_output(corpus, historic_df, already_scored_df, out)
        return pd.read_parquet(out)

    # ── Connect to LLM ────────────────────────────────────────────────────
    try:
        client, model = get_llm_client()
    except ConnectionError as e:
        log.error(f"❌ {e}")
        log.error("  Ensure ngrok is active and LLM_BASE_URL secret is current")
        raise

    records = []
    failed  = 0

    log.info(f"Scoring {len(to_score)} new documents with {model}...")

    for idx, row in to_score:
        text   = str(row.get("text_clean", ""))
        source = str(row.get("source", "UNKNOWN"))
        date   = str(row["date"].date() if hasattr(row["date"], "date") else row["date"])

        scores = score_document_with_retry(client, model, text, source, date)
        if scores.get("llm_failed"):
            failed += 1

        scores = validate_and_clip_scores(scores)
        scores["doc_index"] = idx
        records.append(scores)
        time.sleep(0.3)

        if len(records) % SAVE_EVERY == 0:
            save_incremental(records, out, already_scored_df)
            log.info(f"  Progress: {len(records)}/{len(to_score)} new docs scored | "
                     f"{failed} failed | incremental save ✓")

    save_incremental(records, out, already_scored_df)
    _save_full_output(corpus, historic_df, already_scored_df, out)

    log.info(f"\n✓ LLM scoring complete → {out}")
    log.info(f"  Prompt version:    {PROMPT_VERSION}")
    log.info(f"  New docs scored:   {len(records)} ({len(records)-failed} success, {failed} failed)")

    final = pd.read_parquet(out)
    log.info(f"  Total in output:   {len(final)}")
    log.info(f"\n  Mean scores by source:")
    if "source" in final.columns:
        for src, grp in final.groupby("source"):
            log.info(f"    {src:25s}: "
                     f"oil_impact={grp['oil_impact_score'].mean():+.4f}  "
                     f"divergence={grp['surface_vs_implied_divergence'].mean():.4f}")
    return final


# ── Column definitions ────────────────────────────────────────────────────────
SCORE_COLS = [
    "oil_impact_score", "supply_disruption_signal", "demand_outlook_signal",
    "geopolitical_risk_signal", "surface_vs_implied_divergence",
    "institutional_confidence", "dominant_theme", "reasoning",
    "llm_scored", "llm_failed", "prompt_version",
]

CORPUS_COLS = ["date", "source", "text", "text_clean", "word_count",
               "finbert_score", "finbert_pos", "finbert_neg",
               "finbert_neu", "finbert_confidence"]


def _save_full_output(corpus: pd.DataFrame,
                      historic_df: pd.DataFrame | None,
                      daily_df: pd.DataFrame | None,
                      out: Path):
    """
    Build complete llm_scores.parquet by merging:
    1. Historic scores (matched by date+source key)
    2. Daily scores (matched by corpus index)
    for every document in the current corpus.
    Includes corpus metadata columns (date, source, text_clean) so
    downstream scripts (21 BERTopic, 22 momentum) can read them directly.
    prompt_version column written for every row — historic rows receive
    PROMPT_VERSION_EXPECTED since they were scored with the backfill prompt.
    """
    rows      = []
    daily_idx = set(daily_df.index.tolist()) if daily_df is not None else set()

    hist_lookup = {}
    if historic_df is not None:
        for _, r in historic_df.iterrows():
            key = (str(r["date"].date()), str(r.get("source", "")))
            hist_lookup[key] = r

    for idx, row in corpus.iterrows():
        doc_date   = row["date"]
        doc_source = str(row.get("source", ""))
        hist_key   = (str(doc_date.date()), doc_source)

        if idx in daily_idx and daily_df is not None:
            score_row = daily_df.loc[idx] if idx in daily_df.index else None
        elif hist_key in hist_lookup:
            score_row = hist_lookup[hist_key]
        else:
            score_row = None

        if score_row is not None:
            entry = {col: score_row.get(col, 0.0) for col in SCORE_COLS
                     if col != "prompt_version"}
            # Historic rows may not have prompt_version — backfill with expected hash
            entry["prompt_version"] = score_row.get("prompt_version", PROMPT_VERSION_EXPECTED)
        else:
            entry = neutral_scores("Not scored — no historic or daily score available")

        # Include corpus metadata so script 21 has text_clean available
        for col in CORPUS_COLS:
            if col in row.index:
                entry[col] = row[col]

        entry["doc_index"] = idx
        rows.append(entry)

    result = pd.DataFrame(rows).set_index("doc_index")
    result.to_parquet(out)
    log.info(f"✓ Full output saved: {len(result)} rows → {out}")
    if "text_clean" in result.columns:
        log.info(f"  text_clean column present ✓ ({result['text_clean'].notna().sum()} non-null)")
    else:
        log.warning("  text_clean column MISSING — script 21 will fail")
    if "prompt_version" in result.columns:
        vc = result["prompt_version"].value_counts()
        log.info(f"  Prompt versions in output: {vc.to_dict()}")


if __name__ == "__main__":
    run_llm_scoring()
