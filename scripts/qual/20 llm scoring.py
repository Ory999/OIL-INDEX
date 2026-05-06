"""
Script 20 — LLM Multidimensional Scoring
Model: Locally hosted open-source model via OpenAI-compatible API
       (ngrok tunnel to AI Lab GPU, or Anthropic API fallback)

Outputs: data/raw/llm_scores.parquet

All scores are on a strict -1.0 to +1.0 scale:
  -1.0 = extremely bearish/negative signal for oil prices
   0.0 = neutral / no directional signal
  +1.0 = extremely bullish/positive signal for oil prices

Theoretical link:
  - oil_impact_score:              Direct oil price directional signal
  - supply_disruption_signal:      Supply-side risk (Kilian 2009 supply shock proxy)
  - demand_outlook_signal:         Demand-side outlook (Kilian 2009 demand shock proxy)
  - geopolitical_risk_signal:      Conflict/sanctions risk premium
  - surface_vs_implied_divergence: Information Asymmetry operationalisation (Akerlof 1970)
                                   — gap between stated language and implied reality
  - institutional_confidence:      How certain/committed the institution sounds
"""
import os, json, logging, time, re
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

RAW_DIR    = Path(os.getenv("DATA_DIR",  "data/raw"))
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ── Model configuration ───────────────────────────────────────────────────
# Primary: locally hosted model via ngrok tunnel (AI Lab GPU)
# Fallback: Anthropic API
LLM_BASE_URL  = os.getenv("LLM_BASE_URL",  "http://localhost:11434/v1")
LLM_MODEL     = os.getenv("LLM_MODEL",     "gpt2oss")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_CONF_THRESHOLD = 0.75  # fallback to FinBERT below this confidence

# Source context for prompt calibration
SOURCE_CONTEXT = {
    "OPEC_MOMR":         "OPEC Monthly Oil Market Report — official production cartel assessment",
    "IEA_OMR":           "IEA Oil Market Report — International Energy Agency demand-side assessment",
    "ARAMCO":            "Saudi Aramco official press release — world's largest oil producer",
    "EIA_STEO":          "EIA Short-Term Energy Outlook — US government official oil market forecast",
    "ENERGY_SECRETARY":  "US Energy Secretary official speech — direct US government energy policy signal",
}


# ── Professional system prompt ────────────────────────────────────────────
SYSTEM_PROMPT = """You are a quantitative analyst specialising in WTI crude oil commodity markets with 20 years of experience at a major energy trading desk.

Your task is to analyse official institutional communications and extract structured sentiment signals that quantify their directional impact on WTI crude oil prices.

SCORING FRAMEWORK — ALL SCORES ARE ON A STRICT -1.0 TO +1.0 SCALE:
• -1.0 = Maximum bearish signal (extreme downward pressure on oil prices)
• -0.5 = Moderately bearish
•  0.0 = Neutral / no directional signal
• +0.5 = Moderately bullish
• +1.0 = Maximum bullish signal (extreme upward pressure on oil prices)

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


def get_llm_client():
    """
    Try local model first (AI Lab via ngrok), fall back to Anthropic API.
    Returns (client, model_name, is_local).
    """
    # Try local first
    try:
        client = OpenAI(base_url=LLM_BASE_URL, api_key="none")
        models = client.models.list()
        model  = models.data[0].id if models.data else LLM_MODEL
        log.info(f"✓ Using local model: {model} at {LLM_BASE_URL}")
        return client, model, True
    except Exception as e:
        log.warning(f"  Local model unavailable ({e}) — trying Anthropic API")

    # No Anthropic fallback configured
       log.info("  No Anthropic API key — will use FinBERT fallback")

    log.warning("  No LLM available — will use FinBERT fallback for all documents")
    return None, None, False


def score_document_openai(client, model: str, text: str,
                           source: str, date: str) -> dict | None:
    """Score via OpenAI-compatible API (local model or compatible service)."""
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
        if "```" in raw:
            raw = re.sub(r'```json?|```', '', raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.debug(f"  OpenAI-compatible scoring failed: {e}")
        return None


def score_document_anthropic(client, text: str,
                              source: str, date: str) -> dict | None:
    """Score via Anthropic API (fallback)."""
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user",
                        "content": build_user_prompt(text, source, date)}],
        )
        raw = response.content[0].text.strip()
        if "```" in raw:
            raw = re.sub(r'```json?|```', '', raw).strip()
        return json.loads(raw)
    except Exception as e:
        log.debug(f"  Anthropic scoring failed: {e}")
        return None


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


def finbert_fallback_scores(finbert_row: pd.Series) -> dict:
    """Generate LLM-format scores from FinBERT when LLM is unavailable."""
    fb_score = float(finbert_row.get("finbert_score", 0.0))
    return {
        "oil_impact_score":              fb_score,
        "supply_disruption_signal":      max(0, -fb_score),
        "demand_outlook_signal":         fb_score * 0.5,
        "geopolitical_risk_signal":      abs(fb_score) * 0.4,
        "surface_vs_implied_divergence": 0.0,
        "institutional_confidence":      float(finbert_row.get("finbert_confidence", 0.5)),
        "dominant_theme":                "UNKNOWN",
        "reasoning":                     "FinBERT fallback — LLM unavailable",
        "llm_scored":                    False,
    }


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

    client, model, is_local = get_llm_client()
    records = []

    log.info(f"Scoring {len(corpus)} documents with LLM...")

    for idx, row in corpus.iterrows():
        text   = str(row.get("text_clean", ""))
        source = str(row.get("source", "UNKNOWN"))
        date   = str(row["date"].date() if hasattr(row["date"], "date") else row["date"])

        scores = None

        if client is not None:
            if is_local:
                scores = score_document_openai(client, model, text, source, date)
            else:
                scores = score_document_anthropic(client, text, source, date)

            # Check confidence threshold — fall back to FinBERT if too low
            if scores is not None:
                conf = scores.get("institutional_confidence", 1.0)
                if conf < LLM_CONF_THRESHOLD:
                    log.debug(f"  Low confidence ({conf:.2f}) — blending with FinBERT")
                    fb = finbert_fallback_scores(row)
                    blend = 1.0 - conf
                    scores["oil_impact_score"] = (
                        scores["oil_impact_score"] * (1 - blend)
                        + fb["oil_impact_score"] * blend
                    )

        if scores is None:
            scores = finbert_fallback_scores(row)
        else:
            scores["llm_scored"] = True

        scores = validate_and_clip_scores(scores)
        scores["doc_index"] = idx
        records.append(scores)

        # Rate limiting
        time.sleep(0.3 if is_local else 0.5)

    result_df = pd.DataFrame(records).set_index("doc_index")
    final     = pd.concat([corpus.reset_index(drop=True),
                            result_df.reset_index(drop=True)], axis=1)
    final.to_parquet(out)

    llm_pct = result_df.get("llm_scored", pd.Series(False)).mean() * 100
    log.info(f"✓ LLM scores saved → {out}")
    log.info(f"  LLM scored: {llm_pct:.1f}% | FinBERT fallback: {100 - llm_pct:.1f}%")
    log.info(f"  Mean oil_impact_score by source:")
    for src, grp in final.groupby("source"):
        log.info(f"    {src:25s}: {grp['oil_impact_score'].mean():+.4f}")

    return final


if __name__ == "__main__":
    run_llm_scoring()
