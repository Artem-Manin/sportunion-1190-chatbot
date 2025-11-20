# app_streamlit_pure_llm.py
import os
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# attempt to import the OpenAI new client; app handles missing client gracefully
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

load_dotenv()

# -----------------------
# Config
# -----------------------
DATA_FILE_PATH = Path("data/season_2526.json")   # <- your uploaded file path
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
MAX_CHARS_IN_PROMPT = int(os.getenv("MAX_CHARS_IN_PROMPT", "120000"))  # safety truncation (chars)

# -----------------------
# UI header
# -----------------------
st.set_page_config(page_title="Sport Union 1190 — Football Stat Chatbot", layout="wide")
st.title("Sport Union 1190 — Football Stat Chatbot")
st.markdown(
    """
This app sends a **processed list of players** (small JSON) to the LLM and asks it to answer your question.
We **do not** send the whole raw season file — only the flattened players table — to keep token usage low.
"""
)

if not DATA_FILE_PATH.exists():
    st.error(f"Data file not found at `{DATA_FILE_PATH}`. Put `season_2526.json` at that path or update the path.")
else:
    st.success(f"Loaded data file: `{DATA_FILE_PATH}`")

if OPENAI_API_KEY is None:
    st.warning("OPENAI_API_KEY not set. LLM calls will fail until you add the key to environment or a .env file.")

st.markdown(f"**Model:** `{MODEL}`  •  **Max prompt chars (safety):** {MAX_CHARS_IN_PROMPT}")

# Always show input
question = st.text_input("Ask a question (natural language). Examples: 'How many players do we have?', 'Who is excluded from statistics?'")
ask_btn = st.button("Ask LLM (processed players JSON)")

# -----------------------
# Helpers: load & flatten players
# -----------------------
def load_players_flat(path: Path):
    """Return a Python list of flattened player dicts extracted from the season JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        top = json.load(fh)
    players_json = top.get("response", {}).get("players", [])
    out = []
    for p in players_json:
        out.append({
            "player_id": p.get("baseObjectId"),
            "name": (p.get("name") or "").strip(),
            "exclude_from_statistics": bool(p.get("excludeFromStatistics", False)),
            "created_ts": p.get("createdTimestamp"),
            "updated_ts": p.get("updatedTimestamp"),
        })
    return out

# load once
players_flat = []
if DATA_FILE_PATH.exists():
    try:
        players_flat = load_players_flat(DATA_FILE_PATH)
    except Exception as e:
        st.error(f"Failed to load/flatten players JSON: {e}")

# prepare pretty JSON for LLM (string)
players_json_for_prompt = json.dumps(players_flat, ensure_ascii=False, indent=2)
st.markdown(f"Length: {len(players_json_for_prompt)}")

# If it is too long for safety, it will be truncated later before sending
if len(players_json_for_prompt) > MAX_CHARS_IN_PROMPT:
    st.warning("Processed players JSON is large and will be truncated before sending to the model. "
               "Consider reducing the number of fields or using intent-mode instead.")

# -----------------------
# Robust extractor for different openai client shapes
# -----------------------
def extract_assistant_text(resp):
    """
    Robust extraction of assistant text from OpenAI response objects.
    Works with:
      - resp.choices[0].message.content  (attribute style)
      - resp.choices[0].message["content"] (dict style)
      - resp.choices[0].text (legacy)
      - fallback to str(resp)
    """
    try:
        choice = resp.choices[0]
    except Exception:
        return str(resp)

    # message attribute
    msg = getattr(choice, "message", None)
    if msg is not None:
        # dict-like
        if isinstance(msg, dict):
            if "content" in msg:
                return msg["content"]
        # object with attribute
        if hasattr(msg, "content"):
            return msg.content
        # fallback: str(msg)
        try:
            return str(msg)
        except Exception:
            pass

    # legacy text
    if hasattr(choice, "text"):
        return choice.text

    # dict indexing fallback
    try:
        return choice["message"]["content"]
    except Exception:
        pass

    return str(resp)

# -----------------------
# System instructions and examples
# -----------------------
SYSTEM_INSTRUCTIONS = """
You are a careful data analyst. I will provide a processed list of players as JSON (DATA_START ... DATA_END),
followed by a user's question. You MUST answer using ONLY the provided data. Do NOT hallucinate values.
If the requested information is not present in the data, respond exactly: "I don't see that information in the provided data."
Be concise. When returning lists, you may use JSON arrays or short bullet lists.
"""

EXAMPLES = """
Examples:

Q: "How many players do we have?"
A: "There are X players."  (only if X can be derived from the provided JSON)

Q: "Who is excluded from statistics?"
A: ["Name A", "Name B"]  (only if these names exist in the provided JSON)
"""

# -----------------------
# LLM call (sends processed players JSON, truncated if needed)
# -----------------------
def call_llm_with_players(model, api_key, question, max_chars=MAX_CHARS_IN_PROMPT, max_tokens=800):
    if OpenAI is None:
        raise RuntimeError("OpenAI python client not installed or import failed. Install `openai` package compatible with OpenAI SDK.")
    client = OpenAI(api_key=api_key)

    # prepare JSON chunk (truncate for safety)
    truncated = False
    json_text = players_json_for_prompt
    if len(json_text) > max_chars:
        json_text = json_text[:max_chars]
        truncated = True

    header = (
        "Below is the processed list of players as JSON.\n"
        "Use ONLY this data to answer. If truncated, you may respond 'DATA_TRUNCATED' if you cannot answer reliably.\n\n"
    )

    user_content = header + "DATA_START\n" + json_text + "\nDATA_END\n\n" + f"User question: {question}\n"

    # call the chat completions
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS + "\n" + EXAMPLES},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )

    text = extract_assistant_text(resp).strip()
    return {"text": text, "truncated": truncated, "raw": resp}

# -----------------------
# Button action
# -----------------------
if ask_btn:
    if not question or question.strip() == "":
        st.warning("Please enter a question first.")
    elif not DATA_FILE_PATH.exists():
        st.error("Data file missing; cannot answer.")
    elif OPENAI_API_KEY is None:
        st.error("OPENAI_API_KEY not set. Set it in your environment or a .env file.")
    else:
        with st.spinner("Calling LLM..."):
            try:
                result = call_llm_with_players(MODEL, OPENAI_API_KEY, question)
            except Exception as e:
                st.error(f"LLM call failed: {e}")
            else:
                if result.get("truncated"):
                    st.warning("Note: processed players JSON was truncated before sending to the model. The model may reply 'DATA_TRUNCATED' if it cannot answer from partial data.")
                st.subheader("LLM answer")
                st.text(result["text"])

                with st.expander("Debug: raw LLM response"):
                    try:
                        # try to show sanitized raw resp
                        raw = result.get("raw")
                        # Avoid dumping huge internal objects; show top-level fields
                        out = {}
                        if raw is not None:
                            out["model"] = getattr(raw, "model", None)
                            out["choices_count"] = len(raw.choices) if hasattr(raw, "choices") else None
                            out["truncated_sent"] = result.get("truncated", False)
                            # extract assistant text too
                            out["assistant_text_preview"] = result["text"][:1000]
                        st.json(out)
                    except Exception as e:
                        st.write("Could not render raw response:", e)

# -----------------------
# Footer — show small sample of players and counts
# -----------------------
st.markdown("---")
st.markdown("**Players (sample / stats)**")
col1, col2 = st.columns([2,1])
with col1:
    if players_flat:
        # show first 15 rows as table
        sample = players_flat[:15]
        st.table(sample)
    else:
        st.write("No players loaded.")
with col2:
    st.write(f"Total players loaded: **{len(players_flat)}**")
    excl = [p["name"] for p in players_flat if p.get("exclude_from_statistics")]
    st.write(f"Excluded from statistics: **{len(excl)}**")
    if excl:
        st.write(excl)
