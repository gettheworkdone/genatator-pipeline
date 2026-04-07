from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import zipfile

from flask import Flask, jsonify, request
from pyfaidx import Faidx
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


def _normalize_fasta_text(content: str) -> str:
    fasta_text = (content or "").strip()
    if not fasta_text:
        raise AssertionError("Field DNA sequence or file are required.")
    if not fasta_text.startswith(">"):
        fasta_text = f">sequence\n{fasta_text}\n"
    elif not fasta_text.endswith("\n"):
        fasta_text = f"{fasta_text}\n"
    return fasta_text


def _handle_upload():
    try:
        request_name = f"request_{date.today()}_{datetime.now().microsecond}"

        if "file" in request.files and request.files["file"].filename:
            fasta_text = request.files["file"].read().decode("utf-8")
        else:
            fasta_text = request.form.get("dna")
        fasta_text = _normalize_fasta_text(fasta_text)

        fasta_file_name = f"{request_name}.fa"
        fasta_path = respond_files_path / fasta_file_name
        fasta_path.write_text(fasta_text, encoding="utf-8")

        Faidx(str(fasta_path))
        fai_file_name = f"{fasta_file_name}.fai"

        gff_file_name = f"{request_name}.gff"
        gff_path = respond_files_path / gff_file_name
        pipe(str(fasta_path), output_gff_path=str(gff_path))

        archive_file_name = f"{request_name}_archive.zip"
        archive_path = respond_files_path / archive_file_name
        with zipfile.ZipFile(archive_path, mode="w") as archive:
            archive.write(fasta_path, fasta_file_name)
            archive.write(fasta_path.with_suffix(".fa.fai"), fai_file_name)
            archive.write(gff_path, gff_file_name)

        return jsonify(
            {
                "fasta_file": f"{common_path}{fasta_file_name}",
                "fai_file": f"{common_path}{fai_file_name}",
                "gff_file": f"{common_path}{gff_file_name}",
                "archive": f"{common_path}{archive_file_name}",
            }
        )
    except AssertionError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400


@app.post("/api/genatator-pipeline/upload")
def upload_pipeline():
    return _handle_upload()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
