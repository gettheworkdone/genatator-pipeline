from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from Bio import BiopythonWarning, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

try:
    from .genatator_core import TranscriptPrediction
except ImportError:
    from genatator_core import TranscriptPrediction


LOGGER = logging.getLogger(__name__)


@dataclass
class GeneGroup:
    chrom: str
    strand: str
    transcript_type: str
    start: int
    end: int
    transcripts: list[TranscriptPrediction] = field(default_factory=list)


UCSC_HEADER_RE = re.compile(r"^(?P<chrom>[^:]+):(?P<start>[0-9,]+)-(?P<end>[0-9,]+)$")


def parse_fasta_records(fasta_path: str | Path) -> list[SeqRecord]:
    records = list(SeqIO.parse(str(fasta_path), "fasta"))
    if not records:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return records


def resolve_seqid_and_offset(record: SeqRecord, shift: Optional[str | int]) -> tuple[str, int]:
    seqid = record.id
    offset = 0
    if shift is None:
        return seqid, offset

    if isinstance(shift, int):
        return seqid, int(shift)

    shift_str = str(shift)
    if shift_str.upper() == "UCSC":
        match = UCSC_HEADER_RE.match(record.id)
        if match is None:
            raise ValueError(
                f"shift='UCSC' requires FASTA headers like 'chr:start-end'. Got '{record.id}'."
            )
        seqid = match.group("chrom")
        offset = int(match.group("start").replace(",", ""))
        return seqid, offset

    try:
        offset = int(shift_str)
        return seqid, offset
    except ValueError as exc:
        raise ValueError(f"Unsupported shift value: {shift!r}") from exc


def _find_segments_ones(array: np.ndarray) -> list[tuple[int, int]]:
    ones_idx = np.where(array == 1)[0]
    if ones_idx.size == 0:
        return []
    split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
    split_ones_idx = np.split(ones_idx, split_idx)
    return [(int(segment[0]), int(segment[-1]) + 1) for segment in split_ones_idx]


def _exon_mask_to_cds_mask_benchmark(exon_preds: np.ndarray, seq: str, strand: str = "+") -> np.ndarray:
    exon_preds = np.asarray(exon_preds, dtype=np.uint8)

    if len(seq) < 3 or int(np.sum(exon_preds)) < 3:
        return np.zeros_like(exon_preds)

    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
        exon_preds = exon_preds[::-1]

    exon_positions = np.where(exon_preds == 1)[0]
    exon_seq = ''.join(np.array(list(seq.upper()), dtype=object)[exon_positions])

    best_len_aa = 0
    best_nt_start = None
    best_nt_end = None

    for frame in range(3):
        sub_seq = exon_seq[frame:]
        if len(sub_seq) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BiopythonWarning)
            aa_seq = str(Seq(sub_seq).translate(to_stop=False))
        protein_split = aa_seq.split('*')
        aa_seqs = [
            protein + '*' if i < len(protein_split) - 1 else protein
            for i, protein in enumerate(protein_split)
        ]

        aa_start = 0
        for protein in aa_seqs:
            prot_len = len(protein)
            if prot_len == 0:
                continue

            nt_start = frame + aa_start * 3
            nt_end = nt_start + prot_len * 3
            aa_start += prot_len

            if 'M' in protein and '*' in protein:
                m_pos = protein.find('M')
                orf = protein[m_pos:]
                orf_len_aa = len(orf)

                if orf[0] == 'M' and orf[-1] == '*' and orf_len_aa > best_len_aa:
                    best_len_aa = orf_len_aa
                    best_nt_start = nt_start + m_pos * 3
                    best_nt_end = nt_end

    if best_nt_start is None:
        return np.zeros_like(exon_preds)

    cds_mask = np.zeros_like(exon_preds)
    cds_positions = exon_positions[best_nt_start:best_nt_end]
    cds_mask[cds_positions] = 1

    if strand == "-":
        cds_mask = cds_mask[::-1]

    return cds_mask


def infer_cds_with_benchmark_heuristic(
    sequence: str,
    interval_start: int,
    exons: Sequence[tuple[int, int]],
    strand: str,
) -> list[tuple[int, int]]:
    if not exons:
        return []

    exon_mask = np.zeros(len(sequence), dtype=np.uint8)
    for start, end in sorted(exons):
        rel_start = max(0, int(start) - int(interval_start))
        rel_end = min(len(sequence), int(end) - int(interval_start))
        if rel_end > rel_start:
            exon_mask[rel_start:rel_end] = 1

    cds_mask = _exon_mask_to_cds_mask_benchmark(exon_mask, sequence, strand=strand)
    return [
        (start + int(interval_start), end + int(interval_start))
        for start, end in _find_segments_ones(cds_mask)
    ]


def compute_cds_phase_map(cds_segments: Sequence[tuple[int, int]], strand: str) -> dict[tuple[int, int], int]:
    ordered = list(cds_segments) if strand == "+" else list(reversed(cds_segments))
    phase_map: dict[tuple[int, int], int] = {}
    consumed = 0
    for segment in ordered:
        phase = (3 - (consumed % 3)) % 3
        phase_map[segment] = phase
        consumed += segment[1] - segment[0]
    return phase_map


def group_transcripts_into_genes(predictions: Sequence[TranscriptPrediction]) -> list[GeneGroup]:
    groups: list[GeneGroup] = []
    for pred in sorted(predictions, key=lambda p: (p.chrom, p.strand, p.transcript_type, p.start, p.end)):
        if not groups:
            groups.append(
                GeneGroup(
                    chrom=pred.chrom,
                    strand=pred.strand,
                    transcript_type=pred.transcript_type,
                    start=pred.start,
                    end=pred.end,
                    transcripts=[pred],
                )
            )
            continue

        current = groups[-1]
        if (
            pred.chrom == current.chrom
            and pred.strand == current.strand
            and pred.transcript_type == current.transcript_type
            and pred.start <= current.end
        ):
            current.transcripts.append(pred)
            current.start = min(current.start, pred.start)
            current.end = max(current.end, pred.end)
        else:
            groups.append(
                GeneGroup(
                    chrom=pred.chrom,
                    strand=pred.strand,
                    transcript_type=pred.transcript_type,
                    start=pred.start,
                    end=pred.end,
                    transcripts=[pred],
                )
            )
    return groups


def gff3_write_header(handle) -> None:
    handle.write("##gff-version 3\n")


def gff3_write_feature(
    handle,
    seqid: str,
    source: str,
    ftype: str,
    start0: int,
    end0: int,
    strand: str,
    attrs: dict[str, str],
    phase: str = ".",
) -> None:
    start1 = int(start0) + 1
    end1 = int(end0)
    score = "."
    attr_text = ";".join(f"{k}={v}" for k, v in attrs.items())
    handle.write(
        f"{seqid}	{source}	{ftype}	{start1}	{end1}	{score}	{strand}	{phase}	{attr_text}\n"
    )


def write_predictions_to_gff(
    predictions: Sequence[TranscriptPrediction],
    output_gff_path: str | Path,
    source: str = "GENATATOR-PIPELINE",
) -> str:
    output_gff_path = str(output_gff_path)
    Path(output_gff_path).parent.mkdir(parents=True, exist_ok=True)
    groups = group_transcripts_into_genes(predictions)

    with open(output_gff_path, "w", encoding="utf-8") as handle:
        gff3_write_header(handle)
        for gene_idx, group in enumerate(groups, start=1):
            gene_id = f"GENATATOR_gene_{gene_idx:06d}"
            gff3_write_feature(
                handle,
                seqid=group.chrom,
                source=source,
                ftype="gene",
                start0=group.start,
                end0=group.end,
                strand=group.strand,
                attrs={"ID": gene_id, "Name": gene_id},
            )

            ordered_transcripts = sorted(
                group.transcripts,
                key=lambda p: (p.start, p.end, p.transcript_type_score),
            )
            for tx_idx, pred in enumerate(ordered_transcripts, start=1):
                tx_id = f"{gene_id}.t{tx_idx}"
                tx_attrs = {
                    "ID": tx_id,
                    "Parent": gene_id,
                    "lncRNA_probability": f"{pred.transcript_type_score:.6f}",
                }
                gff3_write_feature(
                    handle,
                    seqid=pred.chrom,
                    source=source,
                    ftype=pred.transcript_type,
                    start0=pred.start,
                    end0=pred.end,
                    strand=pred.strand,
                    attrs=tx_attrs,
                )

                for exon_idx, (start, end) in enumerate(sorted(pred.exons), start=1):
                    gff3_write_feature(
                        handle,
                        seqid=pred.chrom,
                        source=source,
                        ftype="exon",
                        start0=start,
                        end0=end,
                        strand=pred.strand,
                        attrs={"ID": f"{tx_id}.exon{exon_idx}", "Parent": tx_id},
                    )

                for intron_idx, (start, end) in enumerate(sorted(pred.introns), start=1):
                    gff3_write_feature(
                        handle,
                        seqid=pred.chrom,
                        source=source,
                        ftype="intron",
                        start0=start,
                        end0=end,
                        strand=pred.strand,
                        attrs={"ID": f"{tx_id}.intron{intron_idx}", "Parent": tx_id},
                    )

                if pred.transcript_type == "mRNA" and pred.cds:
                    phase_map = compute_cds_phase_map(pred.cds, pred.strand)
                    for cds_idx, segment in enumerate(sorted(pred.cds), start=1):
                        gff3_write_feature(
                            handle,
                            seqid=pred.chrom,
                            source=source,
                            ftype="CDS",
                            start0=segment[0],
                            end0=segment[1],
                            strand=pred.strand,
                            attrs={"ID": f"{tx_id}.cds{cds_idx}", "Parent": tx_id},
                            phase=str(phase_map[segment]),
                        )

    return output_gff_path
