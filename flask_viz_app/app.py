import matplotlib
matplotlib.use("Agg")  # Force non-interactive backend
from flask import Flask, render_template, request
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Ensure no GUI backend is loaded
os.environ["MPLBACKEND"] = "Agg"

app = Flask(__name__)

# Define folders
UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")
        if file and file.filename.endswith(".csv"):
            try:
                # Save the uploaded file
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)

                # Load and process CSV with Pandas
                df = pd.read_csv(filepath)
                if df.empty:
                    return render_template("index.html", error="Empty CSV file")

                # Generate visualizations
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                scatter_plot = f"scatter_{timestamp}.png"
                box_plot = f"box_{timestamp}.png"

                # Scatter plot (first two numerical columns)
                num_cols = df.select_dtypes(include=["float64", "int64"]).columns
                if len(num_cols) >= 2:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(df[num_cols[0]], df[num_cols[1]], c="blue", label="Data Points")
                    plt.xlabel(num_cols[0])
                    plt.ylabel(num_cols[1])
                    plt.title(f"{num_cols[0]} vs. {num_cols[1]}")
                    plt.legend()
                    plt.savefig(os.path.join(STATIC_FOLDER, scatter_plot))
                    plt.close()
                else:
                    scatter_plot = None

                # Box plot (numerical column by categorical column)
                cat_cols = df.select_dtypes(include=["object"]).columns
                if len(num_cols) >= 1 and len(cat_cols) >= 1:
                    plt.figure(figsize=(8, 6))
                    sns.boxplot(x=cat_cols[0], y=num_cols[0], data=df)
                    plt.title(f"{num_cols[0]} by {cat_cols[0]}")
                    plt.savefig(os.path.join(STATIC_FOLDER, box_plot))
                    plt.close()
                else:
                    box_plot = None

                # Summary statistics
                summary = df.describe().to_html()

                return render_template(
                    "index.html",
                    message=f"Uploaded and processed {file.filename}",
                    scatter_plot=scatter_plot,
                    box_plot=box_plot,
                    summary=summary
                )
            except Exception as e:
                return render_template("index.html", error=f"Error processing file: {str(e)}")
        else:
            return render_template("index.html", error="Please upload a CSV file")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)  # Disable auto-reload to avoid backend issues