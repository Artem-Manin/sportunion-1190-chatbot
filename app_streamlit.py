# app_streamlit.py

import os
import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Allow importing build_tables.py from src/
sys.path.append("src")

try:
    import build_tables
except Exception as e:
    st.error(f"Failed to import build_tables.py: {e}")
    build_tables = None

# Optional: OpenAI client
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ================================
# Unified Secrets: st.secrets (Cloud) OR .env (local)
# ================================
load_dotenv()


def get_secret(key: str, default=None):
    """
    Unified secret getter:
    - Try Streamlit secrets first (on Streamlit Cloud).
    - If secrets are not configured or key is missing, fall back to env/.env.
    """
    # Try Streamlit secrets (but handle "no secrets" case gracefully)
    try:
        # This will raise StreamlitSecretNotFoundError if secrets.toml not set up
        value = st.secrets[key]
        if value is not None:
            return value
    except Exception:
        pass  # no secrets configured or key not found

    # Fallback: environment / .env
    value = os.getenv(key, default)
    return value if value is not None else default


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
MODEL = get_secret("LLM_MODEL", "gpt-4o-mini")
MAX_CHARS_IN_PROMPT = int(get_secret("MAX_CHARS_IN_PROMPT", "120000"))  # for safety


# ================================
# Streamlit UI Configuration
# ================================
st.set_page_config(page_title="Sport Union 1190 — Chatbot", layout="wide")
st.title("⚽ Sport Union 1190 – Football Statistics Chatbot")

st.markdown(
    """
This chatbot uses **two combined tables** built from season 24/25 and 25/26:

- **player_stats** — per-player statistics + season  
- **matches** — per-match scores + season  

Your question + the filtered tables are sent to the LLM.
Only the provided data may be used — no hallucinations.
"""
)

st.info(f"Using model: `{MODEL}`")

if OPENAI_API_KEY is None:
    st.warning("⚠️ OPENAI_API_KEY not found. Set it in Streamlit Secrets or in `.env` locally.")


# ================================
# Load / Build Tables (cached)
# ================================
@st.cache_data(show_spinner=True)
def load_combined_tables():
    """Call build_tables.main() → returns player_stats_all & matches_all."""
    if build_tables is None:
        return pd.DataFrame(), pd.DataFrame()

    res = build_tables.main()
    p = res.get("player_stats_all", pd.DataFrame())
    m = res.get("matches_all", pd.DataFrame())

    return p, m


with st.spinner("Preparing data from both seasons..."):
    player_stats_all, matches_all = load_combined_tables()

if player_stats_all.empty or matches_all.empty:
    st.error("❌ Could not load tables from build_tables.py.")
else:
    st.success(
        f"Loaded player_stats_all ({len(player_stats_all)} rows) and "
        f"matches_all ({len(matches_all)} rows)."
    )


# ================================
# Filters: Exclude flag & Seasons
# ================================
st.subheader("Filters")

col1, col2 = st.columns(2)

with col1:
    exclude_flag = st.checkbox(
        "Exclude players who are marked as exclude_from_statistics",
        value=True
    )

with col2:
    seasons_available = sorted(player_stats_all["season"].dropna().unique())
    seasons_selected = st.multiselect(
        "Seasons to include:",
        options=seasons_available,
        default=seasons_available
    )

if not seasons_selected:
    seasons_selected = seasons_available


# Apply filters
player_stats = player_stats_all[player_stats_all["season"].isin(seasons_selected)].copy()
matches = matches_all[matches_all["season"].isin(seasons_selected)].copy()

if exclude_flag:
    player_stats = player_stats[player_stats["exclude_from_statistics"] == False]

# ================================
# Ask Question (Before Tables)
# ================================
st.markdown("---")
st.subheader("Ask a question")

question = st.text_input(
    "Example: 'Who is the top scorer?', 'How many matches ended 4:3?', 'Show all players with >5 goals'"
)
btn = st.button("Ask the LLM")


# ================================
# Show Preview Tables
# ================================
st.markdown("---")
st.subheader("Current Data (Preview)")

colA, colB = st.columns(2)

with colA:
    st.markdown("**Player Stats (Top 15)**")
    st.dataframe(player_stats.head(15))

with colB:
    st.markdown("**Matches (Top 15)**")
    st.dataframe(matches.head(15))

st.markdown(
    f"- Rows in player_stats: **{len(player_stats)}**  \n"
    f"- Rows in matches: **{len(matches)}**"
)


# ================================
# LLM Helper: extract text safely
# ================================
def extract_assistant_text(resp):
    """Handles all possible formats of OpenAI responses."""
    try:
        choice = resp.choices[0]
    except:
        return str(resp)

    msg = getattr(choice, "message", None)

    if msg is not None:
        if isinstance(msg, dict):
            return msg.get("content", str(msg))
        if hasattr(msg, "content"):
            return msg.content

    if hasattr(choice, "text"):
        return choice.text

    try:
        return choice["message"]["content"]
    except:
        return str(resp)


# ================================
# Build prompt & LLM call
# ================================
SYSTEM_INSTRUCTIONS = """
You are a careful football data analyst.
You will receive two JSON tables: player_stats and matches.

RULES:
- Use ONLY the provided data.
- Never invent players or matches.
- If something is not answerable, say:
  "I don't see that information in the provided data."
- Be concise.
"""

EXAMPLES = """
Examples:
- "Who is the top scorer?"
- "How many matches ended with 3:2?"
- "List all players with >5 goals."
"""


def build_json_payload(ps: pd.DataFrame, ms: pd.DataFrame):
    payload = {
        "player_stats": ps.to_dict(orient="records"),
        "matches": ms.to_dict(orient="records")
    }
    text = json.dumps(payload, ensure_ascii=False)

    if len(text) > MAX_CHARS_IN_PROMPT:
        return text[: MAX_CHARS_IN_PROMPT - 50] + '..."TRUNCATED"]}', True

    return text, False


def call_llm(question, ps, ms):
    if OpenAI is None:
        raise RuntimeError("OpenAI client not installed.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    tables_json, truncated = build_json_payload(ps, ms)

    user_content = (
        "DATA_START\n"
        + tables_json +
        "\nDATA_END\n\n"
        f"User question: {question}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + EXAMPLES},
            {"role": "user", "content": user_content}
        ]
    )

    return extract_assistant_text(resp), truncated, resp


# ================================
# Run LLM when button pressed
# ================================
if btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Contacting the LLM..."):
            try:
                answer, truncated, raw = call_llm(question, player_stats, matches)
            except Exception as e:
                st.error(f"LLM ERROR: {e}")
            else:
                if truncated:
                    st.warning("⚠️ Data was truncated before sending to model.")

                st.subheader("LLM Answer")
                st.write(answer)

                with st.expander("LLM Debug Info"):
                    st.json({
                        "input_truncated": truncated,
                        "model": MODEL,
                        "first_500_chars": answer[:500]
                    })
