"""
Carbon Footprint Calculator — Extended Edition
Features:
- Transport / Electricity / LPG inputs
- Multi-day tracking (save daily entries)
- Supabase (preferred) or local SQLite fallback
- Matplotlib charts (time series + breakdown)
- GPT-powered recommendations (OpenAI)
- SDG-branded UI (simple CSS)
- Lightweight multi-page navigation using sidebar
"""

from dotenv import load_dotenv
load_dotenv()

import os
import datetime
import io
import math
import sqlite3
from typing import Optional, Dict, Any, List

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

# -------------------- ENV --------------------
# Streamlit Cloud secrets preferred
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
SUPABASE_URL = st.secrets.get("SUPABASE_URL")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY")
# -------------------- OPENAI CLIENT --------------------
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
if OPENAI_AVAILABLE:
    client = OpenAI(api_key=OPENAI_API_KEY)
    
# Optional: Supabase client
try:
    from supabase import create_client, Client as SupabaseClient  # pip install supabase
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

# Optional: OpenAI client
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

# ---------------------------
# CONFIG & CONSTANTS
# ---------------------------
st.set_page_config(page_title="Carbon Footprint Calculator — Extended", page_icon="🌍", layout="wide")

# Emission factors (kg CO2)
EMISSION_FACTORS = {
    "Car (Petrol)": 0.192,
    "Car (Diesel)": 0.171,
    "Motorbike": 0.103,
    "Matatu/Bus": 0.105,
    "Bicycle/Walking": 0.0
}
ELECTRICITY_FACTOR = 0.18  # kg CO2 per kWh
LPG_FACTOR = 3.0  # kg CO2 per kg LPG

# DB table name
TABLE_NAME = "daily_emissions"

# ---------------------------
# CSS / SDG Branding
# ---------------------------
SDG_COLOR = "#2E8B57"  # pleasant green
st.markdown(
    f"""
    <style>
    .sdg-header {{
        background: linear-gradient(90deg, {SDG_COLOR} 0%, #62c370 100%);
        color: white;
        padding: 16px;
        border-radius: 10px;
    }}
    .card {{
        padding: 14px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        background: white;
    }}
    .small {{
        font-size: 0.9rem;
        color: #444;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# DATABASE: Supabase or SQLite fallback
# ---------------------------

def init_supabase():
    """Return a supabase client or None."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    if not SUPABASE_AVAILABLE:
        st.warning("Supabase client library not installed. Install `supabase` to enable Supabase support.")
        return None
    try:
        client = create_client(url, key)
        return client
    except Exception as e:
        st.error(f"Supabase init error: {e}")
        return None

def init_sqlite(db_path="emissions.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            transport_mode TEXT,
            distance REAL,
            electricity REAL,
            lpg REAL,
            transport_emission REAL,
            electricity_emission REAL,
            lpg_emission REAL,
            total_emission REAL,
            notes TEXT
        )
    """)
    conn.commit()
    return conn

supabase = init_supabase()
sqlite_conn = init_sqlite()

# ---------------------------
# DB helper functions
# ---------------------------

def insert_record_local(record: Dict[str, Any]):
    cur = sqlite_conn.cursor()
    cur.execute(f"""
        INSERT INTO {TABLE_NAME} (
            date, transport_mode, distance, electricity, lpg,
            transport_emission, electricity_emission, lpg_emission,
            total_emission, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record["date"], record["transport_mode"], record["distance"],
        record["electricity"], record["lpg"],
        record["transport_emission"], record["electricity_emission"], record["lpg_emission"],
        record["total_emission"], record.get("notes", "")
    ))
    sqlite_conn.commit()
    return cur.lastrowid

def fetch_all_local() -> pd.DataFrame:
    df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME} ORDER BY date ASC", sqlite_conn)
    if df.empty:
        return pd.DataFrame(columns=[
            "id","date","transport_mode","distance","electricity","lpg",
            "transport_emission","electricity_emission","lpg_emission","total_emission","notes"
        ])
    # convert date
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

# Supabase versions
def insert_record_supabase(record: Dict[str, Any]):
    try:
        supabase.table(TABLE_NAME).insert(record).execute()
        return True
    except Exception as e:
        st.error(f"Supabase insert error: {e}")
        return False

def fetch_all_supabase() -> pd.DataFrame:
    try:
        res = supabase.table(TABLE_NAME).select("*").order("date", {"ascending": True}).execute()
        data = res.data or []
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception as e:
        st.error(f"Supabase fetch error: {e}")
        return pd.DataFrame()

def db_insert(record: Dict[str, Any]):
    """Insert using Supabase if available + configured, else local SQLite."""
    if supabase:
        # Supabase expects ISO date string
        rec_copy = record.copy()
        if isinstance(rec_copy.get("date"), (datetime.date, datetime.datetime)):
            rec_copy["date"] = rec_copy["date"].isoformat()
        ok = insert_record_supabase(rec_copy)
        if ok:
            return
        # fallback to local
    insert_record_local(record)

def db_fetch_all() -> pd.DataFrame:
    if supabase:
        df = fetch_all_supabase()
        if df is not None and not df.empty:
            return df
    return fetch_all_local()

# ---------------------------
# EMISSION CALC
# ---------------------------
def compute_emissions(distance: float, transport_mode: str, electricity: float, lpg: float) -> Dict[str, float]:
    # transport emission
    tf = EMISSION_FACTORS.get(transport_mode, 0.0)
    transport_emission = float(distance) * float(tf)
    electricity_emission = float(electricity) * float(ELECTRICITY_FACTOR)
    lpg_emission = float(lpg) * float(LPG_FACTOR)
    total = transport_emission + electricity_emission + lpg_emission
    # round to 4 decimals for storage
    return {
        "transport_emission": round(transport_emission, 6),
        "electricity_emission": round(electricity_emission, 6),
        "lpg_emission": round(lpg_emission, 6),
        "total_emission": round(total, 6)
    }

# ---------------------------
# GPT RECOMMENDATIONS HELPERS
# ---------------------------
def call_gpt_for_recommendations(recent_df: pd.DataFrame, openai_key: Optional[str]) -> str:
    if not OPENAI_AVAILABLE or openai_key is None:
        return "GPT recommendations unavailable — install `openai` package and set OPENAI_API_KEY."
        # Build a short prompt summarizing recent days
    summary = ""
    if recent_df.empty:
        summary = "No previous entries available. Provide general tips for reducing daily carbon footprint in transport, electricity, and cooking."
    else:
        last = recent_df.tail(7)  # last 7 entries
        rows = []
        for i, r in last.iterrows():
            rows.append(f"{r['date']}: total={r['total_emission']} kg (transport={r['transport_emission']}, elec={r['electricity_emission']}, lpg={r['lpg_emission']})")
        summary = "Last entries:\n" + "\n".join(rows) + "\nProvide 5 concise, actionable personalized recommendations to reduce emissions based on the above data. Focus on low-cost, high-impact actions for a user in Kenya."


 # GPT Tips
    if OPENAI_AVAILABLE:
        if st.button("Get GPT Tips"):
            last = df.tail(7)
            summary = "\n".join([f"{r['date']}: {r['total_emission']:.2f} kg" for _,r in last.iterrows()])
            prompt = f"You are a sustainability assistant. Given recent daily CO₂ totals:\n{summary}\nProvide 10 actionable tips for reducing emissions."
            try:
                response = client.responses.create(model=OPENAI_MODEL, input=prompt, max_output_tokens=300)
                st.markdown(response.output_text)
            except Exception as e:
                st.error(f"AI Error: {e}")
    else:
        st.warning("OpenAI not configured.")

# ---------------------------
# PLOTTING
# ---------------------------
def plot_time_series(df: pd.DataFrame) -> bytes:
    """Return PNG bytes of a time series chart."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data to plot yet", ha="center", va="center")
        ax.axis('off')
    else:
        # aggregate by date
        df_plot = df.copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'])
        df_plot = df_plot.sort_values('date')
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df_plot['date'], df_plot['total_emission'], marker='o', linewidth=2)
        ax.set_title("Total daily CO₂ emissions (kg)")
        ax.set_ylabel("kg CO₂")
        ax.set_xlabel("Date")
        ax.grid(alpha=0.2)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def plot_breakdown_bar(df: pd.DataFrame) -> bytes:
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No data to plot yet", ha="center", va="center")
        ax.axis('off')
    else:
        df2 = df.copy()
        df2['date'] = pd.to_datetime(df2['date'])
        df2 = df2.sort_values('date')
        # Show stacked bar of last 14 days
        last = df2.tail(14)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(last['date'], last['transport_emission'], label='Transport')
        ax.bar(last['date'], last['electricity_emission'], bottom=last['transport_emission'], label='Electricity')
        bottoms = last['transport_emission'] + last['electricity_emission']
        ax.bar(last['date'], last['lpg_emission'], bottom=bottoms, label='LPG')
        ax.set_title("Emissions breakdown (last 14 entries)")
        ax.set_ylabel("kg CO₂")
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ---------------------------
# UI PAGES / NAVIGATION
# ---------------------------
def page_home():
    st.markdown('<div class="sdg-header"><h2>🌍 Carbon Footprint Calculator — Extended</h2><div class="small">SDG 13 — Climate Action | Simple, trackable, and personalized</div></div>', unsafe_allow_html=True)
    st.write("Use the **Enter Data** page to submit daily usage. Then visit **History** and **Insights** for charts and AI recommendations.")

def page_enter_data():
    st.header("Enter daily data")
    col1, col2, col3 = st.columns([3, 2, 2])
    today = st.date_input("Select date", value=datetime.date.today())
    with col1:
        distance = st.number_input("Distance travelled today (km)", min_value=0.0, value=0.0, step=0.1)
        transport_mode = st.selectbox("Transport mode", list(EMISSION_FACTORS.keys()))
    with col2:
        electricity = st.number_input("Electricity used today (kWh)", min_value=0.0, value=0.0, step=0.1)
        lpg = st.number_input("LPG used today (kg)", min_value=0.0, value=0.0, step=0.1)
    notes = st.text_area("Optional notes (e.g., reason for travel, special events)", max_chars=400, height=80)

    if st.button("Save entry"):
        em = compute_emissions(distance, transport_mode, electricity, lpg)
        record = {
            "date": today.isoformat(),
            "transport_mode": transport_mode,
            "distance": float(distance),
            "electricity": float(electricity),
            "lpg": float(lpg),
            "transport_emission": em["transport_emission"],
            "electricity_emission": em["electricity_emission"],
            "lpg_emission": em["lpg_emission"],
            "total_emission": em["total_emission"],
            "notes": notes or ""
        }
        db_insert(record)
        st.success(f"Saved. Total emissions: {em['total_emission']:.3f} kg CO₂")

def page_history():
    st.header("History & Multi-day tracker")
    df = db_fetch_all()
    if df.empty:
        st.info("No entries yet. Add data on the 'Enter Data' page.")
        return
    st.dataframe(df[['date','transport_mode','distance','electricity','lpg','total_emission','notes']].sort_values('date', ascending=False), height=300)

    st.markdown("### Charts")
    ts_png = plot_time_series(df)
    br_png = plot_breakdown_bar(df)
    c1, c2 = st.columns(2)
    with c1:
        st.image(ts_png, use_column_width=True)
    with c2:
        st.image(br_png, use_column_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download all data (CSV)", csv, file_name="emissions_data.csv", mime="text/csv")

def page_insights():
    st.header("AI Recommendations & Insights")
    df = db_fetch_all()
    st.markdown("### Quick stats")
    if df.empty:
        st.info("No data yet. Save some entries first to get personalized recommendations.")
    else:
        total = df['total_emission'].sum()
        avg = df['total_emission'].mean()
        st.metric("Total (saved entries) kg CO₂", f"{total:.2f}")
        st.metric("Average per entry (kg CO₂)", f"{avg:.2f}")

    st.markdown("### GPT-powered Recommendations")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        st.warning("Set the OPENAI_API_KEY environment variable to enable GPT recommendations.")
        return
    if not OPENAI_AVAILABLE:
        st.warning("`openai` package not installed; install it to use GPT recommendations.")
        return

    if st.button("Get personalized tips from GPT"):
        with st.spinner("Asking GPT for suggestions..."):
            text = call_gpt_for_recommendations(df, openai_key)
            st.markdown("#### Recommendations")
            st.write(text)

# ---------------------------
# STREAMLIT APP LAYOUT
# ---------------------------
def main():
    pages = {
        "Home": page_home,
        "Enter Data": page_enter_data,
        "History": page_history,
        "Insights": page_insights
    }
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(pages.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Database**")
    if supabase:
        st.sidebar.success("Supabase configured")
        st.sidebar.write("Note: Supabase table name should be `daily_emissions` with matching columns.")
    else:
        st.sidebar.info("Using local SQLite fallback (emissions.db)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Branding**")
    st.sidebar.markdown("SDG 13 • Climate Action 🌍")
    st.sidebar.markdown("Made with ❤️ — modify freely")
    pages[selection]()

if __name__ == "__main__":
    main()


