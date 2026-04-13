from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

try:
    import h5py
except ImportError:
    h5py = None
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
)

try:
    from .configuration_genatator_pipeline import GenatatorPipelineConfig
    from .genatator_core import (
        EDGE_LABELS_DEFAULT,
        REGION_LABELS_DEFAULT,
        SEGMENTATION_LABELS_DEFAULT,
        TranscriptPrediction,
        build_introns_from_exons,
        build_segmentation_result,
        filter_intervals_by_intragenic,
        find_tss_polya_pairs_right_left_only,
        infer_segmentation_tracks_with_rc,
        infer_sequence_classification_score_with_rc,
        infer_token_classification_tracks_with_rc,
        log_track_dict_summary,
        normalize_dna,
        peak_finding,
        resolve_label_names,
        summarize_array,
        write_intervals_to_bed,
    )
    from .gff_utils import (
        infer_cds_with_benchmark_heuristic,
        parse_fasta_records,
        resolve_seqid_and_offset,
        write_predictions_to_gff,
    )
except ImportError:
    from configuration_genatator_pipeline import GenatatorPipelineConfig
    from genatator_core import (
        EDGE_LABELS_DEFAULT,
        REGION_LABELS_DEFAULT,
        SEGMENTATION_LABELS_DEFAULT,
        TranscriptPrediction,
        build_introns_from_exons,
        build_segmentation_result,
        filter_intervals_by_intragenic,
        find_tss_polya_pairs_right_left_only,
        infer_segmentation_tracks_with_rc,
        infer_sequence_classification_score_with_rc,
        infer_token_classification_tracks_with_rc,
        log_track_dict_summary,
        normalize_dna,
        peak_finding,
        resolve_label_names,
        summarize_array,
        write_intervals_to_bed,
    )
    from gff_utils import (
        infer_cds_with_benchmark_heuristic,
        parse_fasta_records,
        resolve_seqid_and_offset,
        write_predictions_to_gff,
    )


_LOGGER = logging.getLogger(__name__)
_MODEL_BATCH_SIZE = 1
_INIT_ONLY_KEYS = {
    "edge_model_path",
    "region_model_path",
    "transcript_type_model_path",
    "segmentation_model_path",
    "torch_dtype",
    "dtype",
}
_STAGE_RC_KEYS = (
    "gene_finding_use_reverse_complement",
    "transcript_type_use_reverse_complement",
    "segmentation_use_reverse_complement",
)


def _setup_logger(logger: Optional[logging.Logger] = None) -> logging.Logger:
    if logger is not None:
        return logger
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger(__name__)



def _resolve_model_dtype(
    dtype_value: Optional[str | torch.dtype],
    device: torch.device,
) -> Optional[torch.dtype]:
    if dtype_value is None:
        return None

    if isinstance(dtype_value, torch.dtype):
        resolved = dtype_value
    else:
        name = str(dtype_value).lower().replace("torch.", "")
        if name == "auto":
            resolved = torch.float16 if device.type == "cuda" else torch.float32
        else:
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
                "float": torch.float32,
            }
            if name not in mapping:
                raise ValueError(f"Unsupported dtype={dtype_value!r}")
            resolved = mapping[name]

    if device.type == "cpu" and resolved in {torch.float16, torch.bfloat16}:
        return torch.float32
    return resolved



def _label_index(label_names: list[str], candidates: list[str], default: Optional[int] = None) -> int:
    normalized = {
        name.lower().replace("_", "").replace("-", "").replace(" ", ""): idx
        for idx, name in enumerate(label_names)
    }
    for candidate in candidates:
        key = candidate.lower().replace("_", "").replace("-", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    if default is not None:
        return default
    raise KeyError(f"Could not find any of {candidates!r} in labels {label_names!r}")


def _safe_record_stem(name: str) -> str:
    allowed = []
    for ch in str(name):
        if ch.isalnum() or ch in {"_", "-", "."}:
            allowed.append(ch)
        else:
            allowed.append("_")
    return "".join(allowed)


def _segment_mean_probabilities(
    segments: list[tuple[int, int]],
    track: np.ndarray,
    interval_start: int,
) -> list[float]:
    if len(track) == 0:
        return [0.0 for _ in segments]
    probs: list[float] = []
    for start, end in segments:
        rel_start = max(0, int(start) - int(interval_start))
        rel_end = min(len(track), int(end) - int(interval_start))
        if rel_end <= rel_start:
            probs.append(0.0)
            continue
        probs.append(float(np.mean(track[rel_start:rel_end])))
    return probs


def _mean_or_zero(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _intron_label_index(label_names: list[str]) -> int:
    return _label_index(label_names, ["intron"], default=2)


class GenatatorPipeline(Pipeline):
    """Custom Hugging Face pipeline for ab initio transcript discovery and annotation.

    Input: path to a FASTA file.
    Output: path to the written GFF file.
    """

    def __init__(
        self,
        model,
        tokenizer=None,
        feature_extractor=None,
        image_processor=None,
        processor=None,
        modelcard=None,
        framework=None,
        task: str = "genatator-pipeline",
        device: Optional[int | str | torch.device] = None,
        dtype: Optional[str | torch.dtype] = None,
        torch_dtype: Optional[str | torch.dtype] = None,
        binary_output: bool = False,
        logger: Optional[logging.Logger] = None,
        **kwargs,
    ) -> None:
        self.logger = _setup_logger(logger)

        config_defaults = (
            model.config.to_runtime_defaults()
            if hasattr(model.config, "to_runtime_defaults")
            else GenatatorPipelineConfig().to_runtime_defaults()
        )

        legacy_use_reverse_complement = kwargs.pop("use_reverse_complement", None)
        self.runtime_defaults = dict(config_defaults)
        custom_init_keys = set(self.runtime_defaults.keys())
        init_overrides = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in custom_init_keys}
        if legacy_use_reverse_complement is not None:
            for key in _STAGE_RC_KEYS:
                init_overrides.setdefault(key, bool(legacy_use_reverse_complement))
        self.runtime_defaults.update(init_overrides)

        effective_dtype = dtype if dtype is not None else torch_dtype
        if effective_dtype is not None:
            self.runtime_defaults["torch_dtype"] = effective_dtype

        super().__init__(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            processor=processor,
            modelcard=modelcard,
            framework=framework,
            task=task,
            device=device,
            dtype=effective_dtype,
            binary_output=binary_output,
            **kwargs,
        )

        self.submodel_dtype = _resolve_model_dtype(
            self.runtime_defaults.get("torch_dtype"),
            self.device,
        )

        self.logger.info("Initializing GENATATOR pipeline on device %s", self.device)
        self._load_stage_models()

    def _load_stage_models(self) -> None:
        self.logger.info("Loading edge model from %s", self.runtime_defaults["edge_model_path"])
        self.edge_tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_defaults["edge_model_path"],
            trust_remote_code=True,
        )
        self.logger.info("Edge tokenizer is_fast=%s", getattr(self.edge_tokenizer, "is_fast", False))
        self.edge_model = AutoModelForTokenClassification.from_pretrained(
            self.runtime_defaults["edge_model_path"],
            trust_remote_code=True,
            torch_dtype=self.submodel_dtype,
        ).to(self.device)
        self.edge_model.eval()
        self.edge_label_names = resolve_label_names(self.edge_model.config, EDGE_LABELS_DEFAULT)
        self.logger.info("Edge label names: %s", self.edge_label_names)

        self.logger.info("Loading region model from %s", self.runtime_defaults["region_model_path"])
        self.region_tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_defaults["region_model_path"],
            trust_remote_code=True,
        )
        self.logger.info("Region tokenizer is_fast=%s", getattr(self.region_tokenizer, "is_fast", False))
        self.region_model = AutoModelForTokenClassification.from_pretrained(
            self.runtime_defaults["region_model_path"],
            trust_remote_code=True,
            torch_dtype=self.submodel_dtype,
        ).to(self.device)
        self.region_model.eval()
        self.region_label_names = resolve_label_names(self.region_model.config, REGION_LABELS_DEFAULT)
        self.logger.info("Region label names: %s", self.region_label_names)
        if not getattr(self.edge_tokenizer, "is_fast", False):
            raise ValueError(
                "The edge-model tokenizer must be fast because gene finding projects BPE-token outputs back to nucleotides."
            )
        if not getattr(self.region_tokenizer, "is_fast", False):
            raise ValueError(
                "The region-model tokenizer must be fast because gene finding projects BPE-token outputs back to nucleotides."
            )

        self.logger.info(
            "Loading transcript-type model from %s",
            self.runtime_defaults["transcript_type_model_path"],
        )
        self.transcript_type_tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_defaults["transcript_type_model_path"],
            trust_remote_code=True,
        )
        self.logger.info(
            "Transcript-type tokenizer is_fast=%s",
            getattr(self.transcript_type_tokenizer, "is_fast", False),
        )
        self.transcript_type_model = AutoModelForSequenceClassification.from_pretrained(
            self.runtime_defaults["transcript_type_model_path"],
            trust_remote_code=True,
            torch_dtype=self.submodel_dtype,
        ).to(self.device)
        self.transcript_type_model.eval()

        self.logger.info("Loading segmentation model from %s", self.runtime_defaults["segmentation_model_path"])
        self.segmentation_tokenizer = AutoTokenizer.from_pretrained(
            self.runtime_defaults["segmentation_model_path"],
            trust_remote_code=True,
        )
        self.logger.info(
            "Segmentation tokenizer is_fast=%s",
            getattr(self.segmentation_tokenizer, "is_fast", False),
        )
        self.segmentation_model = AutoModelForTokenClassification.from_pretrained(
            self.runtime_defaults["segmentation_model_path"],
            trust_remote_code=True,
            torch_dtype=self.submodel_dtype,
        ).to(self.device)
        self.segmentation_model.eval()
        self.segmentation_label_names = resolve_label_names(
            self.segmentation_model.config,
            SEGMENTATION_LABELS_DEFAULT,
        )
        self.logger.info("Segmentation label names: %s", self.segmentation_label_names)
        self.logger.info("Submodel dtype resolved to %s", self.submodel_dtype)
        self.segmentation_idx_exon = _label_index(self.segmentation_label_names, ["exon"])
        self.segmentation_idx_cds = _label_index(self.segmentation_label_names, ["CDS", "cds"])
        self.segmentation_idx_intron = _intron_label_index(self.segmentation_label_names)

        self.edge_idx_tss_plus = _label_index(self.edge_label_names, ["TSS+"], default=0)
        self.edge_idx_tss_minus = _label_index(self.edge_label_names, ["TSS-"], default=1)
        self.edge_idx_polya_plus = _label_index(self.edge_label_names, ["PolyA+"], default=2)
        self.edge_idx_polya_minus = _label_index(self.edge_label_names, ["PolyA-"], default=3)
        self.region_idx_intragenic_plus = _label_index(
            self.region_label_names,
            ["Intragenic+"],
            default=4,
        )
        self.region_idx_intragenic_minus = _label_index(
            self.region_label_names,
            ["Intragenic-"],
            default=5,
        )

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        forward_kwargs = {}
        postprocess_kwargs = {}

        if "output_gff_path" in kwargs:
            postprocess_kwargs["output_gff_path"] = kwargs.pop("output_gff_path")

        legacy_use_reverse_complement = kwargs.pop("use_reverse_complement", None)

        runtime_defaults = getattr(self, "runtime_defaults", {})
        for key in runtime_defaults:
            if key in _INIT_ONLY_KEYS:
                continue
            if key in kwargs:
                forward_kwargs[key] = kwargs.pop(key)

        if legacy_use_reverse_complement is not None:
            for key in _STAGE_RC_KEYS:
                forward_kwargs.setdefault(key, bool(legacy_use_reverse_complement))

        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def _resolve_intermediate_dir(self, fasta_path: str, params: dict[str, Any]) -> Path:
        target = params.get("intermediate_output_dir")
        if target:
            out_dir = Path(str(target))
        else:
            out_dir = Path(fasta_path).resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    def _save_gene_finding_intermediates(
        self,
        out_dir: Path,
        record_stem: str,
        edge_tracks: np.ndarray,
        region_tracks: np.ndarray,
        peaks: np.ndarray,
        pairs,
        filtered,
    ) -> None:
        self.logger.info("Saving intermediate files to %s for %s", out_dir, record_stem)
        np.save(out_dir / f"{record_stem}.edge_tracks.npy", edge_tracks)
        np.save(out_dir / f"{record_stem}.region_tracks.npy", region_tracks)
        np.save(out_dir / f"{record_stem}.peaks.npy", peaks)
        write_intervals_to_bed(pairs, str(out_dir / f"{record_stem}.candidate_intervals.bed"))
        write_intervals_to_bed(filtered, str(out_dir / f"{record_stem}.filtered_intervals.bed"))

        if h5py is None:
            self.logger.warning("h5py is not installed; skipping HDF5 intermediate dump.")
            return

        h5_path = out_dir / f"{record_stem}.gene_finding_debug.h5"
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("edge_tracks", data=edge_tracks, compression="gzip")
            h5.create_dataset("region_tracks", data=region_tracks, compression="gzip")
            h5.create_dataset("peaks", data=peaks, compression="gzip")
            pairs_coords = np.array([[int(iv.start), int(iv.end)] for iv in pairs], dtype=np.int64) if pairs else np.zeros((0, 2), dtype=np.int64)
            filtered_coords = np.array([[int(iv.start), int(iv.end)] for iv in filtered], dtype=np.int64) if filtered else np.zeros((0, 2), dtype=np.int64)
            h5.create_dataset("candidate_interval_coords", data=pairs_coords, compression="gzip")
            h5.create_dataset("filtered_interval_coords", data=filtered_coords, compression="gzip")
            dt = h5py.string_dtype(encoding="utf-8")
            h5.create_dataset("candidate_interval_strands", data=np.array([iv.strand for iv in pairs], dtype=object), dtype=dt)
            h5.create_dataset("filtered_interval_strands", data=np.array([iv.strand for iv in filtered], dtype=object), dtype=dt)
        self.logger.info("Saved HDF5 gene-finding dump to %s", h5_path)

    def preprocess(self, inputs: Any):
        if isinstance(inputs, dict):
            fasta_path = inputs.get("fasta_path") or inputs.get("fasta")
        else:
            fasta_path = inputs
        if fasta_path is None:
            raise ValueError("Input must be a FASTA path string or a dict containing 'fasta_path'.")
        fasta_path = str(fasta_path)
        if not os.path.exists(fasta_path):
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        records = parse_fasta_records(fasta_path)
        return {"fasta_path": fasta_path, "records": records}

    def forward(self, model_inputs, **forward_params):
        return self._forward(model_inputs, **forward_params)

    def _forward(self, model_inputs: dict[str, Any], **forward_params):
        params = dict(self.runtime_defaults)
        params.update(forward_params)

        predictions: list[TranscriptPrediction] = []
        fasta_path = model_inputs["fasta_path"]
        records = model_inputs["records"]

        self.logger.info(
            "Runtime parameters | edge_context=%s | region_context=%s | tx_type_context=%s | seg_context=%s | edge_overlap=%.3f | region_overlap=%.3f | RC(gene/type/seg)=(%s,%s,%s) | dtype=%s | save_intermediate_files=%s",
            params["edge_context_length"],
            params["region_context_length"],
            params["transcript_type_context_length"],
            params["segmentation_context_length"],
            float(params["edge_context_fraction"]),
            float(params["region_context_fraction"]),
            bool(params["gene_finding_use_reverse_complement"]),
            bool(params["transcript_type_use_reverse_complement"]),
            bool(params["segmentation_use_reverse_complement"]),
            str(self.submodel_dtype),
            bool(params.get("save_intermediate_files", False)),
        )

        for record in tqdm(records, desc="FASTA records"):
            seqid, offset = resolve_seqid_and_offset(record, params["shift"])
            sequence = normalize_dna(str(record.seq))
            record_stem = _safe_record_stem(record.id)
            self.logger.info("Processing record %s (%s), length=%d", record.id, seqid, len(sequence))

            edge_tracks = infer_token_classification_tracks_with_rc(
                sequence=sequence,
                tokenizer=self.edge_tokenizer,
                model=self.edge_model,
                label_names=self.edge_label_names,
                window_size=int(params["edge_context_length"]),
                context_fraction=float(params["edge_context_fraction"]),
                average_token_length=float(params["edge_average_token_length"]),
                max_genomic_chunk_ratio=float(params["edge_max_genomic_chunk_ratio"]),
                drop_last=bool(params["edge_drop_last"]),
                gap_token_id=int(params["edge_gap_token_id"]),
                batch_size=_MODEL_BATCH_SIZE,
                device=self.device,
                apply_sigmoid=bool(params["edge_apply_sigmoid"]),
                rc_task="edge",
                use_reverse_complement=bool(params["gene_finding_use_reverse_complement"]),
                progress_desc=f"{seqid} edge model",
                logger=self.logger,
                chunk_log_every=int(params["chunk_log_every"]),
            )

            region_tracks = infer_token_classification_tracks_with_rc(
                sequence=sequence,
                tokenizer=self.region_tokenizer,
                model=self.region_model,
                label_names=self.region_label_names,
                window_size=int(params["region_context_length"]),
                context_fraction=float(params["region_context_fraction"]),
                average_token_length=float(params["region_average_token_length"]),
                max_genomic_chunk_ratio=float(params["region_max_genomic_chunk_ratio"]),
                drop_last=bool(params["region_drop_last"]),
                gap_token_id=int(params["region_gap_token_id"]),
                batch_size=_MODEL_BATCH_SIZE,
                device=self.device,
                apply_sigmoid=bool(params["region_apply_sigmoid"]),
                rc_task="region",
                use_reverse_complement=bool(params["gene_finding_use_reverse_complement"]),
                progress_desc=f"{seqid} region model",
                logger=self.logger,
                chunk_log_every=int(params["chunk_log_every"]),
            )

            X = np.array(
                [
                    edge_tracks[self.edge_idx_tss_plus],
                    edge_tracks[self.edge_idx_polya_plus],
                    edge_tracks[self.edge_idx_tss_minus],
                    edge_tracks[self.edge_idx_polya_minus],
                ]
            )
            peaks = peak_finding(X, params["lp_frac"], params["pk_prom"], params["pk_dist"], params["pk_height"], logger=self.logger)
            pairs = find_tss_polya_pairs_right_left_only(
                peaks,
                chrom_name=seqid,
                window_size=int(params["interval_window_size"]),
                k=int(params["max_pairs_per_seed"]),
                progress_every=int(params["pairing_progress_every"]),
                logger=self.logger,
            )
            self.logger.info("Record %s: %d candidate intervals before region filtering.", seqid, len(pairs))
            filtered = filter_intervals_by_intragenic(
                pairs=pairs,
                intragenic_plus=region_tracks[self.region_idx_intragenic_plus],
                intragenic_minus=region_tracks[self.region_idx_intragenic_minus],
                prob_threshold=float(params["prob_threshold"]),
                zero_fraction_drop_threshold=float(params["zero_fraction_drop_threshold"]),
                logger=self.logger,
            )
            self.logger.info("Record %s: %d candidate intervals after region filtering.", seqid, len(filtered))
            self.logger.info(
                "Edge merged summary by channel | TSS+=%s | PolyA+=%s | TSS-=%s | PolyA-=%s",
                summarize_array(edge_tracks[self.edge_idx_tss_plus]),
                summarize_array(edge_tracks[self.edge_idx_polya_plus]),
                summarize_array(edge_tracks[self.edge_idx_tss_minus]),
                summarize_array(edge_tracks[self.edge_idx_polya_minus]),
            )
            self.logger.info(
                "Region intragenic summary | plus=%s | minus=%s",
                summarize_array(region_tracks[self.region_idx_intragenic_plus]),
                summarize_array(region_tracks[self.region_idx_intragenic_minus]),
            )

            if bool(params.get("save_intermediate_files", False)):
                out_dir = self._resolve_intermediate_dir(fasta_path, params)
                self._save_gene_finding_intermediates(
                    out_dir=out_dir,
                    record_stem=record_stem,
                    edge_tracks=edge_tracks,
                    region_tracks=region_tracks,
                    peaks=peaks,
                    pairs=pairs,
                    filtered=filtered,
                )

            for interval_idx, interval in enumerate(tqdm(filtered, desc=f"{seqid} transcripts", leave=False), start=1):
                interval_sequence = sequence[interval.start : interval.end]
                if not interval_sequence:
                    continue

                if interval_idx <= 10 or interval_idx % 100 == 0:
                    self.logger.info(
                        "Transcript interval %d/%d | %s:%d-%d (%s) | length=%d",
                        interval_idx,
                        len(filtered),
                        seqid,
                        interval.start,
                        interval.end,
                        interval.strand,
                        len(interval_sequence),
                    )

                transcript_type_score = infer_sequence_classification_score_with_rc(
                    sequence=interval_sequence,
                    tokenizer=self.transcript_type_tokenizer,
                    model=self.transcript_type_model,
                    chunk_size=int(params["transcript_type_context_length"]),
                    overlap=0,
                    batch_size=_MODEL_BATCH_SIZE,
                    device=self.device,
                    use_reverse_complement=bool(params["transcript_type_use_reverse_complement"]),
                    logger=self.logger,
                    apply_sigmoid=bool(params["transcript_type_apply_sigmoid"]),
                )
                transcript_type = (
                    "lnc_RNA"
                    if transcript_type_score >= float(params["transcript_type_threshold"])
                    else "mRNA"
                )
                if interval_idx <= 10 or interval_idx % 100 == 0:
                    self.logger.info(
                        "Transcript interval %d | transcript_type_score=%.5f | threshold=%.5f | assigned=%s",
                        interval_idx,
                        float(transcript_type_score),
                        float(params["transcript_type_threshold"]),
                        transcript_type,
                    )

                segmentation_tracks = infer_segmentation_tracks_with_rc(
                    sequence=interval_sequence,
                    tokenizer=self.segmentation_tokenizer,
                    model=self.segmentation_model,
                    chunk_size=int(params["segmentation_context_length"]),
                    overlap=0,
                    batch_size=_MODEL_BATCH_SIZE,
                    device=self.device,
                    use_reverse_complement=bool(params["segmentation_use_reverse_complement"]),
                    logger=self.logger,
                    progress_desc=f"{seqid} segmentation",
                    apply_sigmoid=bool(params["segmentation_apply_sigmoid"]),
                )
                segmentation_result = build_segmentation_result(
                    interval_start=interval.start,
                    sequence=interval_sequence,
                    strand=interval.strand,
                    segmentation_tracks=segmentation_tracks,
                    splice_filter=bool(params["splice_filter"]),
                    is_cds_binary=False,
                )

                exons = segmentation_result.exon_segments
                if interval_idx <= 10 or interval_idx % 100 == 0:
                    self.logger.info(
                        "Transcript interval %d | segmentation exon_count=%d | cds_count=%d",
                        interval_idx,
                        len(segmentation_result.exon_segments),
                        len(segmentation_result.cds_segments),
                    )
                if not exons:
                    self.logger.warning(
                        "Skipping interval %s:%d-%d (%s): segmentation produced no exons.",
                        seqid,
                        interval.start,
                        interval.end,
                        interval.strand,
                    )
                    continue

                if transcript_type == "mRNA":
                    if bool(params["use_cds_heuristic"]):
                        cds = infer_cds_with_benchmark_heuristic(
                            sequence=interval_sequence,
                            interval_start=interval.start,
                            exons=exons,
                            strand=interval.strand,
                        )
                    else:
                        cds = segmentation_result.cds_segments
                else:
                    cds = []

                exon_mean_probs = _segment_mean_probabilities(
                    exons,
                    segmentation_result.tracks[self.segmentation_idx_exon],
                    interval.start,
                )
                cds_mean_probs = _segment_mean_probabilities(
                    cds,
                    segmentation_result.tracks[self.segmentation_idx_cds],
                    interval.start,
                )
                exon_confidence_total = _mean_or_zero(exon_mean_probs)
                cds_confidence_total = _mean_or_zero(cds_mean_probs)
                segmentation_confidence_total = (
                    cds_confidence_total if transcript_type == "mRNA" and cds_mean_probs else exon_confidence_total
                )

                introns = build_introns_from_exons(exons)

                def _shift_segments(segments):
                    return [(start + offset, end + offset) for start, end in segments]

                predictions.append(
                    TranscriptPrediction(
                        chrom=seqid,
                        start=interval.start + offset,
                        end=interval.end + offset,
                        strand=interval.strand,
                        transcript_type=transcript_type,
                        transcript_type_score=float(transcript_type_score),
                        exons=_shift_segments(exons),
                        introns=_shift_segments(introns),
                        cds=_shift_segments(cds),
                        sequence_name=record.id,
                        exon_mean_probs=exon_mean_probs,
                        cds_mean_probs=cds_mean_probs,
                        exon_confidence_total=exon_confidence_total,
                        cds_confidence_total=cds_confidence_total,
                        segmentation_confidence_total=segmentation_confidence_total,
                    )
                )
                if interval_idx <= 10 or interval_idx % 100 == 0:
                    self.logger.info(
                        "Transcript interval %d | final exons=%d introns=%d cds=%d",
                        interval_idx,
                        len(exons),
                        len(introns),
                        len(cds),
                    )

        return {
            "fasta_path": fasta_path,
            "predictions": predictions,
        }

    def postprocess(self, model_outputs: dict[str, Any], output_gff_path: Optional[str] = None) -> str:
        fasta_path = model_outputs["fasta_path"]
        predictions = model_outputs["predictions"]

        if output_gff_path is None:
            output_gff_path = str(Path(fasta_path).with_suffix(".gff"))

        output_gff_path = write_predictions_to_gff(
            predictions=predictions,
            output_gff_path=output_gff_path,
            source="GENATATOR-PIPELINE",
            transcript_coloring_thresholds=self.runtime_defaults.get("transcript_coloring_thresholds", "auto"),
        )
        self.logger.info("Wrote %d transcript annotations to %s", len(predictions), output_gff_path)
        return output_gff_path
