# scripts/plot_responses_pdf_and_html.py
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
from datetime import datetime

# ---- CONFIG ----
EXCEL_PATH = r"C:\Users\pavan\OneDrive\Desktop\DASC CAPSTONE PROJECT\Responses.xlsx"
OUT_DIR = Path("reports/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = OUT_DIR / "Response_Distributions.pdf"
HTML_INDEX = OUT_DIR / "index.html"        # master listing
HTML_COMBINED = OUT_DIR / "Response_Distributions_all.html"  # one big HTML page

# ---- Helpers ----
def slugify(s: str) -> str:
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "", s)
    return s[:80] or "untitled"

# ---- Load ----
df = pd.read_excel(EXCEL_PATH)
df = df.drop(columns=[c for c in df.columns if "Timestamp" in str(c)], errors="ignore")

# Use a clean style for PDF
plt.style.use("seaborn-v0_8-muted")

# ---- Generate PDF (all questions, one file) ----
with PdfPages(PDF_PATH) as pdf:
    # Cover page
    plt.figure(figsize=(9, 6))
    plt.text(
        0.5, 0.6,
        "Aspira AI — Questionnaire Response Visual Summary",
        ha="center", va="center", fontsize=18, weight="bold"
    )
    plt.text(
        0.5, 0.45,
        f"Total Questions: {len(df.columns)}\nTotal Responses: {len(df)}\nGenerated: {datetime.now():%Y-%m-%d %H:%M}",
        ha="center", va="center", fontsize=12
    )
    plt.axis("off")
    pdf.savefig(); plt.close()

    # One page per question
    for col in df.columns:
        series = df[col]
        # if column is entirely empty, skip
        if series.dropna().empty:
            continue

        counts = series.value_counts(dropna=False).sort_values(ascending=False)
        labels = counts.index.astype(str)

        plt.figure(figsize=(10, 5))

        if len(counts) <= 6:
            # Pie chart
            plt.pie(counts, labels=labels, autopct="%1.1f%%", startangle=90)
            plt.title(f"{col} — Response Distribution", fontsize=12)
        else:
            # Bar chart
            counts.plot(kind="bar", color="steelblue", edgecolor="black")
            plt.title(f"{col} — Response Count", fontsize=12)
            plt.xlabel("Responses")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        pdf.savefig()
        plt.close()

print(f"✅ PDF saved: {PDF_PATH}")

# ---- Generate interactive HTML per-question + index + combined ----
links = []
combined_html_parts = [
    "<!doctype html><html><head><meta charset='utf-8'><title>Response Distributions</title></head><body>",
    "<h1>Aspira AI — Interactive Response Distributions</h1>",
    f"<p><strong>Total Questions:</strong> {len(df.columns)} | <strong>Total Responses:</strong> {len(df)}</p>",
    "<hr>"
]

for i, col in enumerate(df.columns, start=1):
    series = df[col]
    if series.dropna().empty:
        continue

    counts = series.value_counts(dropna=False).sort_values(ascending=False).reset_index()
    counts.columns = ["response", "count"]

    # Decide chart type
    if len(counts) <= 6:
        fig = px.pie(
            counts, names="response", values="count",
            title=f"{col} — Response Distribution",
            hole=0.4
        )
    else:
        fig = px.bar(
            counts, x="response", y="count",
            title=f"{col} — Response Count"
        )
        fig.update_layout(xaxis_tickangle=-30)

    # Save individual HTML
    fname = f"q{i:02d}_{slugify(col)}.html"
    out_path = OUT_DIR / fname
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    links.append((col, fname))

    # Append to combined big HTML (inline without full HTML wrapper)
    combined_html_parts.append(f"<h2>{i}. {col}</h2>")
    combined_html_parts.append(fig.to_html(include_plotlyjs=False, full_html=False))

# Finish combined HTML
combined_html_parts.append("</body></html>")
HTML_COMBINED.write_text("\n".join(combined_html_parts), encoding="utf-8")
print(f"✅ Combined HTML saved: {HTML_COMBINED}")

# Write index with links
index_lines = [
    "<!doctype html><html><head><meta charset='utf-8'><title>Reports Index</title></head><body>",
    "<h1>Aspira AI — Reports</h1>",
    f"<p>Generated: {datetime.now():%Y-%m-%d %H:%M}</p>",
    f"<p><a href='{PDF_PATH.name}'>📄 Response_Distributions.pdf</a></p>",
    f"<p><a href='{HTML_COMBINED.name}'>🌐 Response_Distributions_all.html (single page)</a></p>",
    "<h2>Per-Question Interactive Charts</h2>",
    "<ol>",
]
for title, fname in links:
    index_lines.append(f"<li><a href='{fname}'>{title}</a></li>")
index_lines += ["</ol>", "</body></html>"]

HTML_INDEX.write_text("\n".join(index_lines), encoding="utf-8")
print(f"✅ Index saved: {HTML_INDEX}")

print("\n➡️ Open one of these:")
print(f"   start {HTML_INDEX}")
print(f"   start {HTML_COMBINED}")
