import streamlit as st
import pandas as pd
import duckdb
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="CheQ Data Platform",
    page_icon="⚡",
    layout="wide"
)

# -----------------------------
# WARNING (IMPORTANT)
# -----------------------------
st.warning("⚠️ Use only masked or internal data. Do NOT upload raw sensitive financial data.")

# -----------------------------
# DARK UI
# -----------------------------
st.markdown("""
<style>
body { background-color: #0f172a; color: white; }
.main { background-color: #0f172a; }
h1, h2, h3 { color: white; }
.stButton>button {
    background-color: #ff4d4f;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}
section[data-testid="stSidebar"] {
    background-color: #020617;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOGIN (LOCAL BASIC)
# -----------------------------
def login():
    st.title("🔐 CheQ Internal Login")
    email = st.text_input("Company Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if email.endswith("@cheq.com") and password == "cheq123":
            st.session_state["logged_in"] = True
        else:
            st.error("Access denied")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if os.path.exists("cheq_logo.png"):
        st.image("cheq_logo.png", width=60)
    else:
        st.write("⚡")

with col2:
    st.markdown("### ⚡ CheQ Data Platform")

with col3:
    st.markdown("🔒 Local Secure Mode")

# -----------------------------
# SIDEBAR
# -----------------------------
if os.path.exists("cheq_logo.png"):
    st.sidebar.image("cheq_logo.png", width=120)

st.sidebar.markdown("## ⚡ CheQ Dashboard")
file1 = st.sidebar.file_uploader("Dataset 1", type=["csv"])
file2 = st.sidebar.file_uploader("Dataset 2", type=["csv"])

# -----------------------------
# MAIN LOGIC
# -----------------------------
if file1 and file2:

    st.markdown("## 🔍 Data Preview")

    try:
        df1_preview = pd.read_csv(file1, nrows=100)
        df2_preview = pd.read_csv(file2, nrows=100)
    except:
        st.error("Invalid CSV file")
        st.stop()

    c1, c2 = st.columns(2)
    c1.dataframe(df1_preview)
    c2.dataframe(df2_preview)

    # FIX FILE POINTER
    file1.seek(0)
    file2.seek(0)

    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
    except:
        st.error("Error loading full dataset")
        st.stop()

    st.success("✅ Data Loaded")

    # -----------------------------
    # DATA CLEANING
    # -----------------------------
    st.markdown("## 🧹 Data Cleaning")

    trim_spaces = st.checkbox("Trim spaces", True)
    lower_case = st.checkbox("Lowercase text")
    remove_nulls = st.checkbox("Remove null rows")

    def clean_df(df):
        if trim_spaces:
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        if lower_case:
            df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        if remove_nulls:
            df = df.dropna()
        return df

    df1 = clean_df(df1)
    df2 = clean_df(df2)

    # -----------------------------
    # DATA MASKING (IMPORTANT)
    # -----------------------------
    st.markdown("## 🔒 Data Masking")

    enable_masking = st.checkbox("Enable PII Masking", True)

    def mask_data(df):
        for col in df.columns:
            col_lower = col.lower()

            if "phone" in col_lower:
                df[col] = df[col].astype(str).apply(lambda x: x[:2] + "****")

            elif "email" in col_lower:
                df[col] = df[col].astype(str).apply(
                    lambda x: "****@" + x.split("@")[-1] if "@" in x else x
                )

            elif "card" in col_lower:
                df[col] = df[col].astype(str).apply(lambda x: "****" + x[-4:])

        return df

    if enable_masking:
        df1 = mask_data(df1)
        df2 = mask_data(df2)
        st.success("Masking applied")

    # -----------------------------
    # AI MATCHING
    # -----------------------------
    st.markdown("## 🧠 AI Column Matching")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    cols1 = df1.columns.tolist()
    cols2 = df2.columns.tolist()

    emb1 = model.encode(cols1)
    emb2 = model.encode(cols2)

    matches = []

    for i, col1_name in enumerate(cols1):
        sims = cosine_similarity([emb1[i]], emb2)[0]
        idx = sims.argmax()
        score = sims[idx]

        if score > 0.6:
            matches.append((col1_name, cols2[idx], round(score, 2)))

    matches = sorted(matches, key=lambda x: x[2], reverse=True)

    for m in matches[:5]:
        st.write(f"🔹 {m[0]} ↔ {m[1]} ({m[2]})")

    default_col1, default_col2, _ = matches[0]

    use_ai = st.checkbox("Use AI Suggestion", True)

    if use_ai:
        join_col1 = default_col1
        join_col2 = default_col2
    else:
        join_col1 = st.selectbox("Column 1", cols1)
        join_col2 = st.selectbox("Column 2", cols2)

    # -----------------------------
    # COLUMN VALIDATION
    # -----------------------------
    st.markdown("## ⚠️ Join Validation")

    allowed_columns = ["id", "user_id", "customer_id", "transaction_id"]

    if join_col1.lower() not in allowed_columns or join_col2.lower() not in allowed_columns:
        st.warning("⚠️ Non-standard join column selected. Verify correctness.")

    # -----------------------------
    # DUPLICATES
    # -----------------------------
    st.markdown("## 🔍 Duplicate Check")

    d1, d2 = st.columns(2)
    d1.metric("Duplicates Dataset 1", df1.duplicated(subset=[join_col1]).sum())
    d2.metric("Duplicates Dataset 2", df2.duplicated(subset=[join_col2]).sum())

    # -----------------------------
    # JOIN
    # -----------------------------
    join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])

    if st.button("🚀 Run Join"):

        con = duckdb.connect()
        con.register("df1", df1)
        con.register("df2", df2)

        query = f"""
        SELECT *
        FROM df1
        {join_type.upper()} JOIN df2
        ON df1.{join_col1} = df2.{join_col2}
        """

        result = con.execute(query).fetch_df()

        st.success("Join Completed")

        # -----------------------------
        # INSIGHTS
        # -----------------------------
        st.markdown("## 📊 Insights")

        total = len(result)
        matched = result[join_col1].notnull().sum()
        unmatched = total - matched

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Rows", total)
        c2.metric("Matched", matched)
        c3.metric("Unmatched", unmatched)

        # -----------------------------
        # OUTPUT
        # -----------------------------
        st.dataframe(result.head(100))

        st.download_button(
            "Download CSV",
            result.to_csv(index=False),
            "joined_data.csv"
        )
