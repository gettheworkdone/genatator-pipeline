from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline(
    task="genatator-pipeline",
    model="shmelev/genatator-pipeline",
    trust_remote_code=True,
    device=0,
)


@app.post("/run")
def run_pipeline():
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "'file' is required"}), 400

    output_gff_path = request.form.get("output_gff_path")
    if not output_gff_path:
        return jsonify({"status": "error", "message": "'output_gff_path' is required"}), 400

    with tempfile.TemporaryDirectory(prefix="genatator_") as tmp_dir:
        fasta_path = Path(tmp_dir) / "input.fasta"
        request.files["file"].save(fasta_path)
        output_path = pipe(str(fasta_path), output_gff_path=output_gff_path)

    return jsonify({"output_gff_path": output_path})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
