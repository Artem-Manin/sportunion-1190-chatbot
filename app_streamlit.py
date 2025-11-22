# app_streamlit.py

import os
import sys
import json
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
    try:
        value = st.secrets[key]
        if value is not None:
            return value
    except Exception:
        pass

    value = os.getenv(key, default)
    return value if value is not None else default


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
MODEL = get_secret("LLM_MODEL", "gpt-4.1-mini")
MAX_CHARS_IN_PROMPT = int(get_secret("MAX_CHARS_IN_PROMPT", "1000000"))  # for safety


# ================================
# Streamlit UI Configuration
# ================================
st.set_page_config(page_title="Sport Union 1190 — Chatbot", layout="wide")

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1rem !important;
        }
        h1 {
            margin-top: 0rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚽ Sport Union 1190 – Football Statistics Chatbot")

st.markdown("Chat with statistics from seasons 24/25 and 25/26 (players, matches, goals).")

if OPENAI_API_KEY is None:
    st.warning("⚠️ OPENAI_API_KEY not found. Set it in Streamlit Secrets or in `.env` locally.")


# ================================
# Cached loader for combined tables
# ================================
@st.cache_data(show_spinner=True)
def load_combined_tables(download_2526: bool):
    """
    Load combined tables.
    - download_2526=True → download & refresh 25/26
    - download_2526=False → use local files only
    """
    if build_tables is None:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # call your updated build_tables main()
    res = build_tables.main(download_2526=download_2526)

    player_stats_all = res.get("player_stats_all", pd.DataFrame())
    matches_all = res.get("matches_all", pd.DataFrame())
    events_all = res.get("events_all", pd.DataFrame())

    return player_stats_all, matches_all, events_all


# ================================
# Data status + Refresh button (sidebar)
# ================================
with st.sidebar:
    st.subheader("Data status")

    refresh_clicked = st.button("🔄 Refresh 25/26 data")

    st.caption(
        """
        Season 24/25 is always loaded locally.  
        Season 25/26 is refreshed from the API when you press the button.
        """
    )

    # Clear cache so refresh loads immediately
    if refresh_clicked:
        try:
            load_combined_tables.clear()
        except Exception:
            st.cache_data.clear()

    with st.spinner("Loading data..."):
        player_stats_all, matches_all, events_all = load_combined_tables(
            download_2526=refresh_clicked
        )

    if player_stats_all.empty or matches_all.empty:
        st.error("❌ Could not load tables.")
    else:
        dates = pd.to_datetime(matches_all["item_event_date"], errors="coerce").dropna()
        last_game = dates.max().strftime("%d.%m.%Y") if len(dates) else "unknown"
        st.caption(f"📅 Last game: **{last_game}**")


# ================================
# Seasons available
# ================================
if not player_stats_all.empty:
    seasons_available = sorted(player_stats_all["season"].dropna().unique().tolist())
else:
    seasons_available = []


# ================================
# Filters UI (BEFORE question)
# ================================
with st.expander("⚙️ Filters", expanded=False):

    # Defaults for first run
    default_exclude = st.session_state.get("exclude_flag", True)
    if seasons_available:
        default_seasons = st.session_state.get("seasons_selected", seasons_available)
        # ensure valid
        default_seasons = [s for s in default_seasons if s in seasons_available] or seasons_available
    else:
        default_seasons = []

    col1, col2 = st.columns(2)

    with col1:
        st.checkbox(
            "Exclude players who are marked as exclude_from_statistics",
            value=default_exclude,
            key="exclude_flag",
        )

    with col2:
        st.multiselect(
            "Seasons to include:",
            options=seasons_available,
            default=default_seasons,
            key="seasons_selected",
        )

    if seasons_available and not st.session_state.get("seasons_selected"):
        st.caption("No season selected – all seasons will be used automatically.")


# Read final filter values from session_state
exclude_flag = st.session_state.get("exclude_flag", True)
if seasons_available:
    seasons_selected = st.session_state.get("seasons_selected", seasons_available)
    seasons_selected = [s for s in seasons_selected if s in seasons_available] or seasons_available
else:
    seasons_selected = []


# ================================
# Build filtered tables based on filters
# ================================
if not player_stats_all.empty:
    player_stats = player_stats_all[player_stats_all["season"].isin(seasons_selected)].copy()
else:
    player_stats = pd.DataFrame()

if not matches_all.empty:
    matches = matches_all[matches_all["season"].isin(seasons_selected)].copy()
else:
    matches = pd.DataFrame()

if not events_all.empty:
    events = events_all[events_all["season"].isin(seasons_selected)].copy()
else:
    events = pd.DataFrame()

# Apply "exclude_from_statistics" flag only to player_stats
if (
    exclude_flag
    and not player_stats.empty
    and "exclude_from_statistics" in player_stats.columns
):
    player_stats = player_stats[player_stats["exclude_from_statistics"] == False]

# Keep only events that belong to the filtered matches (via match_key, if present)
if (
    not events.empty
    and not matches.empty
    and "match_key" in events.columns
    and "match_key" in matches.columns
):
    events = events.merge(
        matches[["season", "match_key"]].drop_duplicates(),
        on=["season", "match_key"],
        how="inner",
    )


# ================================
# Ask Question — form so Enter submits
# ================================
with st.form("question_form"):
    question = st.text_input(
        "Example: 'Who is the top scorer?', 'How many matches ended in a draw?', "
        "'Show all players with >5 goals', 'Show all goals in the last match with minute, scorer and assist.'"
    )
    submitted = st.form_submit_button("Ask your question")


# ================================
# LLM Helper: extract text safely
# ================================
def extract_assistant_text(resp):
    """Handles all possible formats of OpenAI responses."""
    try:
        choice = resp.choices[0]
    except Exception:
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
    except Exception:
        return str(resp)


# ================================
# Build prompt & LLM call
# ================================
SYSTEM_INSTRUCTIONS = """
You are a careful football data analyst.
You will receive three JSON tables: player_stats, matches and events.

TABLES:
- player_stats: per-player statistics by season
- matches: per match, with score and metadata
- events: one row per goal (time, scorer, assist, own goal, match reference)

RULES:
- Use ONLY the provided data for anything factual.
- You ARE allowed to propose groupings or divisions of players into teams, even if such teams do not exist in the raw data.
- When creating teams, distribute players fairly based on available statistics.
- If something is factually missing (goals, matches, etc.), say so.
- Be concise.
- Ignore the shortened surname (e.g., “Artem M.”) and treat it as the full first name (Artem), unless there are multiple players with the same first name.
"""

EXAMPLES = """
Examples:
- "Who is the top scorer?"
- "How many matches ended in a draw?"
- "List all players with >5 goals."
- "Show all goals in the last match with minute, scorer and assist."
"""


def build_json_payload(ps: pd.DataFrame, ms: pd.DataFrame, ev: pd.DataFrame):
    payload = {
        "player_stats": ps.to_dict(orient="records"),
        "matches": ms.to_dict(orient="records"),
        "events": ev.to_dict(orient="records"),
    }
    text = json.dumps(payload, ensure_ascii=False)

    if len(text) > MAX_CHARS_IN_PROMPT:
        # naive truncation to stay under limit
        return text[: MAX_CHARS_IN_PROMPT - 50] + '..."TRUNCATED"]}', True

    return text, False


def call_llm(question, ps, ms, ev):
    if OpenAI is None:
        raise RuntimeError("OpenAI client not installed.")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    tables_json, truncated = build_json_payload(ps, ms, ev)

    user_content = (
        "DATA_START\n"
        + tables_json
        + "\nDATA_END\n\n"
        f"User question: {question}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.0,
        max_tokens=1200,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + EXAMPLES},
            {"role": "user", "content": user_content},
        ],
    )

    return extract_assistant_text(resp), truncated, resp


# ================================
# Run LLM when form submitted
# ================================
if submitted:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        if player_stats.empty or matches.empty:
            st.warning("No data available for the selected filters.")
        else:
            with st.spinner("Processing your question..."):
                try:
                    answer, truncated, raw = call_llm(question, player_stats, matches, events)
                except Exception as e:
                    st.error(f"Error while contacting the model: {e}")
                else:
                    st.subheader("Answer")
                    st.write(answer)

                    if truncated:
                        st.warning("⚠️ Data was truncated before sending to the model.")


# ================================
# Sample questions
# ================================
with st.expander("💬 Sample questions", expanded=False):
    st.markdown(
        "- How well did Uli play?\n"
        "- Who are Karl’s main competitors this season for goals and assists, both in terms of total numbers and average per game?\n"
        "- Using only the available statistical data, divide the players into two balanced teams "
        "by distributing different performance profiles evenly between both sides.\n"
        "- Does Artem tend to score earlier or later in matches?\n"
        "- How effective is the partnership between Engin and Peter? Have they combined for many goals?\n"
        "- How many comebacks did we have this season?"
    )


# ================================
# Show Preview Tables
# ================================
with st.expander("🔍 Data preview", expanded=False):

    st.subheader("Player Stats (Top 5)")
    st.dataframe(player_stats.head(5))
    st.markdown(f"Rows in player_stats: **{len(player_stats)}**")

    st.subheader("Matches (Top 5)")
    st.dataframe(matches.head(5))
    st.markdown(f"Rows in matches: **{len(matches)}**")

    st.subheader("Events (Top 5)")
    st.dataframe(events.head(5))
    st.markdown(f"Rows in events: **{len(events)}**")


# ================================
# Prompt Size Debug (Bottom Expander)
# ================================
with st.expander("🧪 Prompt size", expanded=False):
    try:
        # Convert tables individually
        ps_json = json.dumps(player_stats.to_dict(orient="records"), ensure_ascii=False)
        ms_json = json.dumps(matches.to_dict(orient="records"), ensure_ascii=False)
        ev_json = json.dumps(events.to_dict(orient="records"), ensure_ascii=False)

        ps_size = len(ps_json)
        ms_size = len(ms_json)
        ev_size = len(ev_json)

        # combined full JSON BEFORE truncation
        full_json = json.dumps(
            {
                "player_stats": player_stats.to_dict(orient="records"),
                "matches": matches.to_dict(orient="records"),
                "events": events.to_dict(orient="records"),
            },
            ensure_ascii=False,
        )
        full_size = len(full_json)

        # JSON AFTER truncation (sent to LLM)
        sent_json, was_truncated = build_json_payload(player_stats, matches, events)
        sent_size = len(sent_json)

        st.subheader("📊 Size per table")
        st.write(f"🟦 player_stats: {ps_size:,} characters")
        st.write(f"🟩 matches:      {ms_size:,} characters")
        st.write(f"🟧 events:       {ev_size:,} characters")

        st.subheader("📦 Combined JSON")
        st.write(f"📦 Original JSON size: {full_size:,} characters")
        st.write(f"✉️ Sent JSON size:     {sent_size:,} characters")
        st.write(f"📉 Max allowed size:   {MAX_CHARS_IN_PROMPT:,} characters")
        st.write(f"🔢 Approx tokens sent: {sent_size / 4:,.0f}")

        if was_truncated:
            st.error("⚠️ JSON WAS TRUNCATED before sending to the model!")
        else:
            st.success("✅ Full JSON sent (no truncation).")

    except Exception as e:
        st.error(f"Error calculating JSON sizes: {e}")
