from flask import Flask, render_template, request
from cleaner import clean_data
from models import run_models
import pandas as pd
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    target = request.form["target"]
    problem_type = request.form["problem_type"]

    filepath = os.path.jo in(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(filepath, encoding='cp1252')
    except Exception as e:
        return f"<h2>Error reading file</h2><p>{str(e)}</p>"

    # Clean data
    df = clean_data(df, target)

    # Run models
    results = run_models(df, target, problem_type)

    # Build results HTML
    best = results[0]
    html = f"<h2>Results for {file.filename}</h2>"
    html += f"<p>Best model: <strong>{best['Model']}</strong></p>"
    html += "<table border='1' cellpadding='8'><tr><th>Model</th><th>Score</th></tr>"
    for r in results:
        score = list(r.values())[1]
        html += f"<tr><td>{r['Model']}</td><td>{score}</td></tr>"
    html += "</table>"
    html += "<br><a href='/'>Upload another file</a>"

    return html

if __name__ == "__main__":
    app.run(debug=True)