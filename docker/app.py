from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)
respond_files_path = Path("/generated/genatator-pipeline")
respond_files_path.mkdir(parents=True, exist_ok=True)
common_path = "/generated/genatator-pipeline/"

pipe = pipeline(
    task="genatator-pipeline",
    model="shmelev/genatator-pipeline",
    trust_remote_code=True,
    device=0,
)


def _handle_upload():
    try:
        request_name = f"request_{date.today()}_{datetime.now().microsecond}"

        with tempfile.TemporaryDirectory(prefix="genatator_") as tmp_dir:
            fasta_path = Path(tmp_dir) / "input.fasta"

            if "file" in request.files and request.files["file"].filename:
                request.files["file"].save(fasta_path)
            else:
                fasta_content = request.form.get("dna")
                if not fasta_content:
                    raise AssertionError("Field DNA sequence or file are required.")
                fasta_text = fasta_content.strip()
                if not fasta_text:
                    raise AssertionError("Field DNA sequence or file are required.")
                if not fasta_text.startswith(">"):
                    fasta_text = f">sequence\n{fasta_text}\n"
                elif not fasta_text.endswith("\n"):
                    fasta_text = f"{fasta_text}\n"
                fasta_path.write_text(fasta_text, encoding="utf-8")

            output_file_name = f"{request_name}.gff"
            output_gff_path = respond_files_path / output_file_name
            output_path = pipe(str(fasta_path), output_gff_path=str(output_gff_path))

        return jsonify({"gff_file": f"{common_path}{Path(output_path).name}"})
    except AssertionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.post("/api/genatator-pipeline/upload")
def upload_pipeline():
    return _handle_upload()


@app.post("/run")
def run_pipeline_backward_compat():
    return _handle_upload()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
