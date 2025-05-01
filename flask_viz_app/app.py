from flask import Flask, render_template, request
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No file selected")
        if file and file.filename.endswith(".csv"):
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            return render_template("index.html", message=f"Uploaded {file.filename}")
        else:
            return render_template("index.html", error="Please upload a CSV file")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)