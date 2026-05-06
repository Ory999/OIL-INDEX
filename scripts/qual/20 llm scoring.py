"""
Script 20 — LLM Multidimensional Scoring
Model: Locally hosted openai/gpt-oss-20b via ngrok tunnel

All documents scored by LLM only. If a document fails after retries
it receives neutral scores (0.0) rather than FinBERT substitution.
FinBERT scores remain available as a separate baseline column for
academic comparison but do NOT replace LLM scores.

All scores on strict -1.0 to +1.0 scale.
"""
import os, json, logging, time, re
from pathlib import Path
import pandas as pd
import numpy as np
import httpx
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR = Path(os.getenv("DATA_DIR", "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

LLM_BASE_URL       = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL          = os.getenv("LLM_MODEL",    "openai/gpt-oss-20b")
LLM_CONF_THRESHOLD = 0.75
MAX_RETRIES        = 3
RETRY_DELAY        = 5.0   # seconds between retries

SOURCE_CONTEXT = {
    "OPEC_MOMR":        "OPEC Monthly Oil Market Report — official production cartel assessment",
    "IEA_OMR":          "IEA Oil Market Report — International Energy Agency demand-side assessment",
    "ARAMCO":           "Saudi Aramco official press release — world's largest oil producer",
    "EIA_STEO":         "EIA Short-Term Energy Outlook — US government official oil market forecast",
    "ENERGY_SECRETARY": "US Energy Secretary official speech — direct US government energy policy signal",
}

SYSTEM_PROMPT = """You are a quantitative analyst specialising in WTI crude oil commodity markets with 20 years of experience at a major energy trading desk.

Your task is to analyse official institutional communications and extract structured sentiment signals that quantify their directional impact on WTI crude oil prices.

SCORING FRAMEWORK — ALL SCORES ARE ON A STRICT -1.0 TO +1.0 SCALE:
- -1.0 = Maximum bearish signal (extreme downward pressure on oil prices)
- -0.5 = Moderately bearish
-  0.0 = Neutral / no directional signal
- +0.5 = Moderately bullish
- +1.0 = Maximum bullish signal (extreme upward pressure on oil prices)

SCORE DEFINITIONS:
1. oil_impact_score (-1 to +1):
   Consider: production cuts (+), supply disruptions (+), demand growth (+)
   vs. production increases (-), demand weakness (-), oversupply (-), price caps (-)

2. supply_disruption_signal (-1 to +1):
   +1 = severe imminent supply disruption (war, sanctions, OPEC cut, pipeline failure)
   -1 = large supply surplus, production ramp-up, supply glut
   0  = balanced supply conditions

3. demand_outlook_signal (-1 to +1):
   +1 = strong demand growth expected (economic expansion, emerging market growth)
   -1 = demand destruction expected (recession, EV transition, efficiency gains)
   0  = stable demand outlook

4. geopolitical_risk_signal (-1 to +1):
   +1 = high geopolitical risk in oil-producing regions (conflict, sanctions, instability)
   -1 = geopolitical de-escalation, stability improvement, sanctions relief
   0  = no significant geopolitical developments

5. surface_vs_implied_divergence (0 to 1, NOT negative):
   Measures the GAP between surface language and implied market reality.
   HIGH (near 1.0) when: institution uses neutral/reassuring language but implies bearish reality,
   OR uses positive language but implies risk. Classic examples:
   — "The market remains broadly balanced" when supply data shows tightening = high divergence
   — "We are monitoring the situation carefully" before announcing sanctions = high divergence
   — "Production levels are appropriate" when hinting at cuts = high divergence
   LOW (near 0) when surface language directly matches the implied market signal.

6. institutional_confidence (0 to 1):
   How certain and committed the institution sounds.
   1.0 = definitive statements, clear commitments, specific numbers
   0.5 = hedged language, conditional statements
   0.0 = vague, non-committal, placeholder language

CRITICAL RULES:
— Return ONLY valid JSON, no other text, no markdown
— All scores except surface_vs_implied_divergence must be between -1.0 and +1.0
— surface_vs_implied_divergence must be between 0.0 and 1.0
— Base scores on the OIL MARKET IMPACT, not general economic sentiment
— Consider the historical precedent: what typically happens to oil prices after similar statements?
— High surface_vs_implied_divergence scores should be rare (reserved for genuinely ambiguous rhetoric)"""


def build_user_prompt(text: str, source: str, date: str) -> str:
    source_context = SOURCE_CONTEXT.get(source, f"Official institutional statement — {source}")
    text_excerpt   = text[:2000].strip()
    return f"""DOCUMENT ANALYSIS REQUEST

Source type: {source_context}
Publication date: {date}
Text excerpt (first 2000 characters):
\"\"\"
{text_excerpt}
\"\"\"

Analyse this document for its directional impact on WTI crude oil prices.
Consider:
1. What explicit statements are made about supply, demand, and prices?
2. What is IMPLIED but not directly stated? (key for divergence score)
3. What historical precedent exists for similar institutional language?
4. How does the source's authority affect market credibility?

Return ONLY this JSON object with no other text:
{{
    "oil_impact_score":              <float -1.0 to 1.0>,
    "supply_disruption_signal":      <float -1.0 to 1.0>,
    "demand_outlook_signal":         <float -1.0 to 1.0>,
    "geopolitical_risk_signal":      <float -1.0 to 1.0>,
    "surface_vs_implied_divergence": <float 0.0 to 1.0>,
    "institutional_confidence":      <float 0.0 to 1.0>,
    "dominant_theme":                <string: one of SUPPLY_CONCERN|DEMAND_WEAKNESS|GEOPOLITICAL|PRODUCTION_CUT|PRODUCTION_INCREASE|MARKET_BALANCE|PRICE_FORECAST|SANCTIONS|NEUTRAL>,
    "reasoning":                     <string: max 50 words explaining the scores>
}}"""


def neutral_scores(reason: str = "LLM failed after retries") -> dict:
    """
    Return neutral 0.0 scores when LLM fails after all retries.
    These are clearly marked as failed so they can be filtered
    or imputed in downstream analysis.
    """
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
    }


def get_llm_client():
    """Connect to locally hosted model via ngrok tunnel."""
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
        return client, model
    except Exception as e:
        raise ConnectionError(f"Cannot connect to local model at {LLM_BASE_URL}: {e}")


def score_document_with_retry(client, model: str, text: str,
                               source: str, date: str) -> dict:
    """
    Score a document with retry logic.
    Tries MAX_RETRIES times before returning neutral scores.
    Never falls back to FinBERT — neutral 0.0 is preferable to
    a misrepresented classification score.
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
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown if model wraps response
            if "```" in raw:
                raw = re.sub(r'```json?|```', '', raw).strip()

            scores = json.loads(raw)
            scores["llm_scored"] = True
            scores["llm_failed"] = False
            return scores

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}"
            log.debug(f"  Attempt {attempt}/{MAX_RETRIES} — {last_error}")
            # Give model a moment then retry
            time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error = str(e)
            log.debug(f"  Attempt {attempt}/{MAX_RETRIES} — LLM call failed: {e}")
            time.sleep(RETRY_DELAY)

    log.warning(f"  All {MAX_RETRIES} attempts failed — using neutral scores ({last_error[:60]})")
    return neutral_scores(f"Failed after {MAX_RETRIES} retries: {last_error[:80]}")


def validate_and_clip_scores(scores: dict) -> dict:
    """Ensure all scores are within valid ranges."""
    clip_fields = [
        "oil_impact_score", "supply_disruption_signal",
        "demand_outlook_signal", "geopolitical_risk_signal",
    ]
    for field in clip_fields:
        if field in scores:
            scores[field] = float(np.clip(scores[field], -1.0, 1.0))

    if "surface_vs_implied_divergence" in scores:
        scores["surface_vs_implied_divergence"] = float(
            np.clip(scores["surface_vs_implied_divergence"], 0.0, 1.0)
        )
    if "institutional_confidence" in scores:
        scores["institutional_confidence"] = float(
            np.clip(scores["institutional_confidence"], 0.0, 1.0)
        )
    return scores


def run_llm_scoring():
    finbert_path = RAW_DIR / "finbert_scores.parquet"
    out          = RAW_DIR / "llm_scores.parquet"

    if not finbert_path.exists():
        log.warning("finbert_scores.parquet not found — run 19 finbert scoring.py first")
        return pd.DataFrame()

    corpus = pd.read_parquet(finbert_path)
    if len(corpus) == 0:
        log.warning("FinBERT corpus empty — skipping LLM scoring")
        return pd.DataFrame()

    # Connect to LLM — raise immediately if unavailable
    # (no silent fallback to FinBERT)
    try:
        client, model = get_llm_client()
    except ConnectionError as e:
        log.error(f"❌ {e}")
        log.error("  LLM is required — cannot proceed without local model")
        log.error("  Ensure ngrok tunnel is active and LLM_BASE_URL secret is current")
        raise

    records     = []
    failed      = 0
    total       = len(corpus)

    log.info(f"Scoring {total} documents with {model}...")
    log.info(f"  Max retries per document: {MAX_RETRIES}")
    log.info(f"  Neutral scores assigned on failure (no FinBERT substitution)")

    for idx, row in corpus.iterrows():
        text   = str(row.get("text_clean", ""))
        source = str(row.get("source", "UNKNOWN"))
        date   = str(row["date"].date() if hasattr(row["date"], "date") else row["date"])

        scores = score_document_with_retry(client, model, text, source, date)

        if scores.get("llm_failed"):
            failed += 1

        scores = validate_and_clip_scores(scores)
        scores["doc_index"] = idx
        records.append(scores)

        # Polite rate limiting
        time.sleep(0.3)

        # Progress log every 50 documents
        if (idx + 1) % 50 == 0:
            log.info(f"  Progress: {idx + 1}/{total} documents "
                     f"({failed} failed so far)")

    result_df = pd.DataFrame(records).set_index("doc_index")
    final     = pd.concat([corpus.reset_index(drop=True),
                            result_df.reset_index(drop=True)], axis=1)
    final.to_parquet(out)

    success_pct = (total - failed) / total * 100
    log.info(f"\n✓ LLM scoring complete → {out}")
    log.info(f"  Total documents:    {total}")
    log.info(f"  LLM scored:         {total - failed} ({success_pct:.1f}%)")
    log.info(f"  Neutral (failed):   {failed} ({100 - success_pct:.1f}%)")
    log.info(f"\n  Mean scores by source:")
    for src, grp in final.groupby("source"):
        scored = grp[grp["llm_failed"] == False] if "llm_failed" in grp else grp
        log.info(f"    {src:25s}: oil_impact={grp['oil_impact_score'].mean():+.4f}  "
                 f"divergence={grp['surface_vs_implied_divergence'].mean():.4f}")

    return final


if __name__ == "__main__":
    run_llm_scoring()
