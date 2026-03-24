from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import duckdb
import pandas as pd
import os
from difflib import SequenceMatcher

app = FastAPI(title="AI Data Join API")

# -----------------------------
# ENABLE CORS (IMPORTANT)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DB CONNECTION
# -----------------------------
def get_connection():
    con = duckdb.connect()

    try:
        con.execute(f"PRAGMA threads={os.cpu_count()}")
    except:
        pass

    try:
        con.execute("PRAGMA enable_object_cache")
    except:
        pass

    try:
        con.execute("PRAGMA memory_limit='8GB'")
    except:
        pass

    return con

con = get_connection()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def fast_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_overlap(col1, col2):
    try:
        s1 = con.execute(f'SELECT "{col1}" FROM t1 LIMIT 200').df()[col1]
        s2 = con.execute(f'SELECT "{col2}" FROM t2 LIMIT 200').df()[col2]

        overlap = len(set(s1).intersection(set(s2)))
        return overlap / min(len(s1), len(s2)) if min(len(s1), len(s2)) > 0 else 0
    except:
        return 0

# -----------------------------
# LOAD FILES
# -----------------------------
@app.post("/load")
async def load_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        path1 = f"temp_{file1.filename}"
        path2 = f"temp_{file2.filename}"

        with open(path1, "wb") as f:
            f.write(await file1.read())

        with open(path2, "wb") as f:
            f.write(await file2.read())

        con.execute(f"CREATE OR REPLACE VIEW t1 AS SELECT * FROM read_csv_auto('{path1}')")
        con.execute(f"CREATE OR REPLACE VIEW t2 AS SELECT * FROM read_csv_auto('{path2}')")

        return {"message": "Files loaded successfully"}

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# PREVIEW DATA
# -----------------------------
@app.get("/preview")
def preview():
    try:
        df1 = con.execute("SELECT * FROM t1 LIMIT 20").df().to_dict()
        df2 = con.execute("SELECT * FROM t2 LIMIT 20").df().to_dict()

        return {
            "table1": df1,
            "table2": df2
        }
    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# SUGGEST JOIN
# -----------------------------
@app.get("/suggest_join")
def suggest_join():
    try:
        preview1 = con.execute("SELECT * FROM t1 LIMIT 20").df()
        preview2 = con.execute("SELECT * FROM t2 LIMIT 20").df()

        cols1 = preview1.columns.tolist()[:20]
        cols2 = preview2.columns.tolist()[:20]

        recommendations = []

        for c1 in cols1:
            for c2 in cols2:
                name_score = fast_similarity(c1, c2)
                dtype_score = 1 if str(preview1[c1].dtype) == str(preview2[c2].dtype) else 0
                overlap_score = get_overlap(c1, c2)

                final_score = (
                    0.5 * name_score +
                    0.2 * dtype_score +
                    0.3 * overlap_score
                )

                recommendations.append((c1, c2, round(final_score, 2)))

        recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)

        if not recommendations:
            return {"error": "No match found"}

        best = recommendations[0]

        return {
            "recommended_join": {
                "column1": best[0],
                "column2": best[1],
                "confidence": best[2]
            }
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# RUN JOIN
# -----------------------------
@app.get("/run_join")
def run_join(col1: str, col2: str, join_type: str = "INNER"):
    try:
        query = f"""
        SELECT *
        FROM t1
        {join_type} JOIN t2
        ON t1."{col1}" = t2."{col2}"
        """

        output_file = "joined_output.parquet"

        con.execute(f"COPY ({query}) TO '{output_file}' (FORMAT PARQUET);")

        return {
            "message": "Join completed successfully",
            "output_file": output_file
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# BASIC METRICS
# -----------------------------
@app.get("/metrics")
def metrics():
    try:
        total = con.execute("SELECT COUNT(*) FROM t1").fetchone()[0]
        return {"rows_in_dataset1": total}
    except Exception as e:
        return {"error": str(e)}
