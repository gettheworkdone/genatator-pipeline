from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
import torch
from scipy.signal import find_peaks
from tqdm.auto import tqdm


LOGGER = logging.getLogger(__name__)


EDGE_LABELS_DEFAULT = ["TSS+", "TSS-", "PolyA+", "PolyA-"]
REGION_LABELS_DEFAULT = ["TSS+", "TSS-", "PolyA+", "PolyA-", "Intragenic+", "Intragenic-"]
SEGMENTATION_LABELS_DEFAULT = ["5UTR", "exon", "intron", "3UTR", "CDS"]

LEFT_SET = {"AG", "AC", "TG"}
RIGHT_SET = {"GT", "GC", "AT"}
INTRON_IDX = 2


@dataclass
class TranscriptInterval:
    chrom: str
    start: int
    end: int
    strand: str
    extra: list[str]


@dataclass
class SegmentationResult:
    exon_segments: list[tuple[int, int]]
    cds_segments: list[tuple[int, int]]
    labels_argmax: np.ndarray
    tracks: np.ndarray


@dataclass
class TranscriptPrediction:
    chrom: str
    start: int
    end: int
    strand: str
    transcript_type: str
    transcript_type_score: float
    exons: list[tuple[int, int]]
    introns: list[tuple[int, int]]
    cds: list[tuple[int, int]]
    sequence_name: str
    exon_mean_probs: list[float] = field(default_factory=list)
    cds_mean_probs: list[float] = field(default_factory=list)
    exon_confidence_total: float = 0.0
    cds_confidence_total: float = 0.0
    segmentation_confidence_total: float = 0.0


def batched(items: Sequence, batch_size: int) -> Iterator[Sequence]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def normalize_dna(seq: str) -> str:
    seq = seq.upper().replace("U", "T")
    return "".join(base if base in {"A", "C", "G", "T", "N"} else "N" for base in seq)


def reverse_complement(seq: str) -> str:
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(table)[::-1]


def infer_reasonable_context_length(tokenizer, model, fallback: int) -> int:
    candidates: list[int] = []
    for obj in (getattr(tokenizer, "model_max_length", None), getattr(model.config, "max_position_embeddings", None)):
        if isinstance(obj, int) and 0 < obj < 10_000_000:
            candidates.append(obj)
    if not candidates:
        return int(fallback)
    return int(min([fallback] + candidates))


def _normalize_label(label: str) -> str:
    return label.lower().replace("_", "").replace(" ", "")


def _is_generic_hf_label(label: str) -> bool:
    normalized = label.strip().lower().replace("-", "_")
    if normalized in {"label", "labels"}:
        return True
    if normalized.startswith("label_") and normalized[6:].isdigit():
        return True
    if normalized.startswith("class_") and normalized[6:].isdigit():
        return True
    if normalized.startswith("score_") and normalized[6:].isdigit():
        return True
    return False


def resolve_label_names(config, fallback: Sequence[str]) -> list[str]:
    id2label = getattr(config, "id2label", None)
    if isinstance(id2label, dict) and len(id2label) == len(fallback):
        ordered = [id2label.get(i) if i in id2label else id2label.get(str(i)) for i in range(len(fallback))]
        if all(isinstance(x, str) and x for x in ordered):
            if any(_is_generic_hf_label(x) for x in ordered):
                return list(fallback)
            return list(ordered)
    return list(fallback)


def build_channel_permutation(label_names: Sequence[str], task: str) -> list[int]:
    normalized = {_normalize_label(name): idx for idx, name in enumerate(label_names)}
    if task in {"edge", "region"}:
        mapping = {
            "tss+": "tss-",
            "tss-": "tss+",
            "polya+": "polya-",
            "polya-": "polya+",
            "intragenic+": "intragenic-",
            "intragenic-": "intragenic+",
        }
    elif task == "segmentation":
        mapping = {
            "5utr": "3utr",
            "3utr": "5utr",
            "exon": "exon",
            "intron": "intron",
            "cds": "cds",
        }
    else:
        raise ValueError(f"Unknown task for channel permutation: {task}")

    permutation: list[int] = []
    for name in label_names:
        key = _normalize_label(name)
        source_name = mapping.get(key, key)
        if source_name not in normalized:
            raise KeyError(f"Could not map RC channel '{source_name}' for labels {label_names}")
        permutation.append(normalized[source_name])
    return permutation


def build_overlap_window_plan(length: int, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    window_size = max(1, int(window_size))
    if length <= window_size:
        return [(0, length)]
    overlap = max(0, min(int(overlap), window_size - 1))
    step = max(1, window_size - overlap)

    starts = list(range(0, max(1, length - window_size + 1), step))
    last_start = max(0, length - window_size)
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, min(length, start + window_size)) for start in starts]


def _offsets_to_numpy(offsets) -> np.ndarray:
    if isinstance(offsets, torch.Tensor):
        return offsets.detach().cpu().numpy()
    return np.asarray(offsets)


def token_scores_to_base_tracks(
    token_scores: np.ndarray,
    offsets: np.ndarray,
    sequence_length: int,
    num_labels: int,
) -> np.ndarray:
    result = np.zeros((num_labels, sequence_length), dtype=np.float32)
    counts = np.zeros(sequence_length, dtype=np.float32)

    for token_idx, span in enumerate(offsets):
        start = int(span[0])
        end = int(span[1])
        if end <= start:
            continue
        start = max(0, min(start, sequence_length))
        end = max(0, min(end, sequence_length))
        if end <= start:
            continue
        result[:, start:end] += token_scores[token_idx].T[:, None]
        counts[start:end] += 1.0

    valid = counts > 0
    if np.any(valid):
        result[:, valid] /= counts[valid][None, :]
    return result


def _extract_special_mask_from_encoding(encoding, tokenizer) -> np.ndarray:
    special = encoding.get("special_tokens_mask")
    if special is None:
        input_ids = encoding["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        special = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    if isinstance(special, torch.Tensor):
        special = special.detach().cpu().numpy()
    special = np.asarray(special)
    if special.ndim > 1:
        special = special[0]
    return special.astype(bool)


def _select_model_inputs_from_encoding(encoding, model, device: torch.device) -> dict:
    accepted_args = set(inspect.signature(model.forward).parameters)
    model_inputs = {}
    for key, value in encoding.items():
        if key in {"offset_mapping", "special_tokens_mask"}:
            continue
        if key not in accepted_args:
            continue
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(device)
        else:
            model_inputs[key] = value
    return model_inputs


def _extract_prediction_tensor(outputs, preferred_key: str = "predicts"):
    if isinstance(outputs, dict):
        if preferred_key in outputs:
            return outputs[preferred_key]
        if "logits" in outputs:
            return outputs["logits"]
    if hasattr(outputs, preferred_key):
        return getattr(outputs, preferred_key)
    if hasattr(outputs, "logits"):
        return outputs.logits
    raise AttributeError(f"Could not extract prediction tensor from output type {type(outputs)!r}")


def _project_token_scores_without_offsets(
    token_scores: np.ndarray,
    sequence_length: int,
    num_labels: int,
) -> np.ndarray:
    result = np.zeros((num_labels, sequence_length), dtype=np.float32)
    token_scores = np.asarray(token_scores, dtype=np.float32)
    if token_scores.size == 0 or sequence_length <= 0:
        return result

    token_count = int(token_scores.shape[0])
    if token_count == sequence_length:
        return token_scores.T.astype(np.float32, copy=False)

    boundaries = np.rint(np.linspace(0, sequence_length, token_count + 1)).astype(int)
    boundaries[0] = 0
    boundaries[-1] = sequence_length
    for token_idx in range(token_count):
        start = int(boundaries[token_idx])
        end = int(boundaries[token_idx + 1])
        if end <= start:
            continue
        result[:, start:end] += token_scores[token_idx][:, None]
    return result


def _canonical_export_label(label_name: str) -> str:
    label = label_name.strip().lower().replace(" ", "").replace("_", "")
    if label.startswith("tss") and label.endswith("+"):
        return "tss_+"
    if label.startswith("tss") and label.endswith("-"):
        return "tss_-"
    if label.startswith("polya") and label.endswith("+"):
        return "polya_+"
    if label.startswith("polya") and label.endswith("-"):
        return "polya_-"
    if label.startswith("intragenic") and label.endswith("+"):
        return "intragenic_+"
    if label.startswith("intragenic") and label.endswith("-"):
        return "intragenic_-"
    raise KeyError(f"Unsupported edge/region label name: {label_name!r}")


def summarize_array(arr: np.ndarray) -> dict[str, float]:
    arr = np.asarray(arr, dtype=np.float32)
    return {
        "min": float(np.min(arr)) if arr.size else 0.0,
        "mean": float(np.mean(arr)) if arr.size else 0.0,
        "max": float(np.max(arr)) if arr.size else 0.0,
        "p95": float(np.quantile(arr, 0.95)) if arr.size else 0.0,
        "nz@0.5": int(np.count_nonzero(arr > 0.5)),
    }


def log_track_dict_summary(
    logger: Optional[logging.Logger],
    prefix: str,
    tracks: dict[str, np.ndarray],
) -> None:
    logger = logger or LOGGER
    for key, values in tracks.items():
        stats = summarize_array(values)
        logger.info(
            "%s | %s | min=%.5f mean=%.5f p95=%.5f max=%.5f n(>0.5)=%d",
            prefix,
            key,
            stats["min"],
            stats["mean"],
            stats["p95"],
            stats["max"],
            stats["nz@0.5"],
        )


def write_intervals_to_bed(
    intervals: Sequence[TranscriptInterval],
    bed_path: str,
) -> str:
    with open(bed_path, "w", encoding="utf-8") as handle:
        for interval in intervals:
            handle.write(
                f"{interval.chrom}\t{int(interval.start)}\t{int(interval.end)}\t{interval.strand}\n"
            )
    return bed_path


def _build_dataset_style_chunk_plan(
    sequence_length: int,
    model_input_length_token: int,
    average_token_length: float,
    chunk_overlap: float,
    max_genomic_chunk_ratio: float,
    drop_last: bool,
) -> tuple[list[tuple[int, int]], int, int]:
    chunk_length = max(1, int(model_input_length_token * average_token_length))
    genomic_chunk_length = max(chunk_length, int(chunk_length * max_genomic_chunk_ratio))

    shift = chunk_length - int(chunk_overlap * chunk_length)
    shift = max(shift, 1)
    step_size = shift

    chunks: list[tuple[int, int]] = []
    if drop_last:
        num_chunks = max(1, (sequence_length - shift) // step_size)
    else:
        num_chunks = max(1, sequence_length // step_size)

    end_pos = 0
    for chunk_idx in range(num_chunks):
        start_pos = chunk_idx * step_size
        end_pos = min(start_pos + chunk_length, sequence_length)
        chunks.append((start_pos, end_pos))

    if (not drop_last) and end_pos < sequence_length:
        chunks.append((end_pos, sequence_length))

    if not chunks:
        chunks = [(0, sequence_length)]
    return chunks, chunk_length, genomic_chunk_length


def _tokenize_dataset_style_window(
    sequence: str,
    tokenizer,
    model_input_length_token: int,
    gap_token_id: Optional[int],
):
    return_token_type_ids = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
    encoding = tokenizer(
        sequence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=int(model_input_length_token),
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=return_token_type_ids,
        verbose=False,
    )
    offsets = _offsets_to_numpy(encoding["offset_mapping"])
    if offsets.ndim > 2:
        offsets = offsets[0]
    input_ids = encoding["input_ids"]
    if isinstance(input_ids, torch.Tensor):
        input_ids = input_ids[0].detach().cpu().tolist()
    if gap_token_id is not None:
        for idx, token_id in enumerate(input_ids):
            if idx > 0 and int(token_id) == int(gap_token_id):
                offsets[idx][0] = offsets[idx - 1][1]
    return encoding, offsets


def _run_token_classification_window(
    encoding,
    model,
    device: torch.device,
    apply_sigmoid: bool,
) -> np.ndarray:
    model_inputs = _select_model_inputs_from_encoding(encoding, model, device)
    outputs = model(**model_inputs)
    scores = _extract_prediction_tensor(outputs, preferred_key="predicts").detach().float()[0]
    if apply_sigmoid:
        scores = scores.sigmoid()
    return scores.cpu().numpy()


def _exporter_trim_bounds(num_tokens: int) -> tuple[int, int]:
    if num_tokens <= 2:
        return 0, num_tokens
    return 1, num_tokens - 1


def _init_export_track_dict(keys: Sequence[str], sequence_length: int) -> dict[str, np.ndarray]:
    tracks = {}
    for key in keys:
        arr = np.empty((sequence_length,), dtype=np.float32)
        arr[:] = np.nan
        tracks[key] = arr
    return tracks


def _infer_exporter_style_tracks_single_strand(
    sequence: str,
    tokenizer,
    model,
    label_names: Sequence[str],
    model_input_length_token: int,
    average_token_length: float,
    chunk_overlap: float,
    max_genomic_chunk_ratio: float,
    drop_last: bool,
    gap_token_id: Optional[int],
    device: torch.device,
    apply_sigmoid: bool,
    reference_strand: str,
    progress_desc: str,
    logger: Optional[logging.Logger] = None,
    chunk_log_every: int = 1000,
) -> dict[str, np.ndarray]:
    logger = logger or LOGGER
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "Edge and region inference require a fast tokenizer with offset mappings for BPE-to-nucleotide projection."
        )
    if reference_strand not in {"+", "-"}:
        raise ValueError(f"reference_strand must be '+' or '-', got {reference_strand!r}")

    canonical_keys = [_canonical_export_label(name) for name in label_names]
    suffix = "" if reference_strand == "+" else "rev_comp_"
    output_keys = [f"{key}{suffix}" for key in canonical_keys]
    sequence_length = len(sequence)
    tracks = _init_export_track_dict(output_keys, sequence_length)

    chunk_plan, chunk_length, genomic_chunk_length = _build_dataset_style_chunk_plan(
        sequence_length=sequence_length,
        model_input_length_token=int(model_input_length_token),
        average_token_length=float(average_token_length),
        chunk_overlap=float(chunk_overlap),
        max_genomic_chunk_ratio=float(max_genomic_chunk_ratio),
        drop_last=bool(drop_last),
    )
    extension = max(0, genomic_chunk_length - chunk_length)
    step_size = max(1, chunk_length - int(chunk_overlap * chunk_length))
    logger.info(
        "%s | strand=%s | seq_len=%d | token_context=%d | avg_token_len=%.3f | chunk_bp=%d | genomic_chunk_bp=%d | step_bp=%d | chunks=%d | drop_last=%s | gap_token_id=%s",
        progress_desc,
        reference_strand,
        sequence_length,
        int(model_input_length_token),
        float(average_token_length),
        int(chunk_length),
        int(genomic_chunk_length),
        int(step_size),
        len(chunk_plan),
        bool(drop_last),
        str(gap_token_id),
    )

    with torch.no_grad():
        for chunk_idx, (start_pos, end_pos) in enumerate(tqdm(chunk_plan, desc=progress_desc, total=len(chunk_plan), leave=False), start=1):
            extended_end = min(sequence_length, end_pos + extension)
            window_sequence = sequence[start_pos:extended_end]
            if not window_sequence:
                continue
            if reference_strand == "-":
                window_sequence = reverse_complement(window_sequence)

            encoding, offsets = _tokenize_dataset_style_window(
                sequence=window_sequence,
                tokenizer=tokenizer,
                model_input_length_token=int(model_input_length_token),
                gap_token_id=gap_token_id,
            )
            score_np = _run_token_classification_window(
                encoding=encoding,
                model=model,
                device=device,
                apply_sigmoid=apply_sigmoid,
            )

            trim_start, trim_end = _exporter_trim_bounds(score_np.shape[0])
            trimmed_scores = score_np[trim_start:trim_end]
            trimmed_offsets = offsets[trim_start:trim_end]
            usable = min(trimmed_scores.shape[0], len(trimmed_offsets))
            trimmed_scores = trimmed_scores[:usable]
            trimmed_offsets = trimmed_offsets[:usable]

            if chunk_log_every and (chunk_idx == 1 or chunk_idx % int(chunk_log_every) == 0 or chunk_idx == len(chunk_plan)):
                usable_bp = int(np.max(trimmed_offsets[:, 1])) if len(trimmed_offsets) else 0
                logger.info(
                    "%s | chunk=%d/%d | genomic_window=[%d,%d) | fetched_end=%d | usable_tokens=%d | usable_bp=%d | token_scores_max=%.5f",
                    progress_desc,
                    chunk_idx,
                    len(chunk_plan),
                    start_pos,
                    end_pos,
                    extended_end,
                    usable,
                    usable_bp,
                    float(np.max(trimmed_scores)) if trimmed_scores.size else 0.0,
                )

            for token_index, span in enumerate(trimmed_offsets):
                token_start = int(span[0])
                token_end = int(span[1])
                if token_end <= token_start:
                    continue
                if reference_strand == "-":
                    genome_start = extended_end - token_end
                    genome_end = extended_end - token_start
                else:
                    genome_start = start_pos + token_start
                    genome_end = start_pos + token_end

                genome_start = max(0, min(genome_start, sequence_length))
                genome_end = max(0, min(genome_end, sequence_length))
                if genome_end <= genome_start:
                    continue

                for class_idx, out_key in enumerate(output_keys):
                    current = tracks[out_key][genome_start:genome_end]
                    value = float(trimmed_scores[token_index, class_idx])
                    if current.size == 0:
                        continue
                    if np.isnan(current).any():
                        tracks[out_key][genome_start:genome_end] = value
                    else:
                        tracks[out_key][genome_start:genome_end] = (float(np.mean(current)) + value) / 2.0

    for values in tracks.values():
        np.nan_to_num(values, copy=False, nan=0.0)
    log_track_dict_summary(logger, f"{progress_desc} summary", tracks)
    return tracks


def _merge_forward_and_rc_track_dicts(
    label_names: Sequence[str],
    forward_tracks: dict[str, np.ndarray],
    reverse_tracks: Optional[dict[str, np.ndarray]],
) -> np.ndarray:
    canonical_keys = [_canonical_export_label(name) for name in label_names]
    if not canonical_keys:
        raise ValueError("No label names provided for track merging.")

    if reverse_tracks is None:
        return np.stack([np.asarray(forward_tracks[key], dtype=np.float32) for key in canonical_keys], axis=0)

    merged: dict[str, np.ndarray] = {}
    stems = sorted({key.rsplit("_", 1)[0] for key in canonical_keys})
    template = np.zeros_like(next(iter(forward_tracks.values())), dtype=np.float32)

    for stem in stems:
        forward_plus = np.asarray(forward_tracks.get(f"{stem}_+", template), dtype=np.float32)
        forward_minus = np.asarray(forward_tracks.get(f"{stem}_-", template), dtype=np.float32)
        reverse_plus = np.asarray(reverse_tracks.get(f"{stem}_+rev_comp_", template), dtype=np.float32)
        reverse_minus = np.asarray(reverse_tracks.get(f"{stem}_-rev_comp_", template), dtype=np.float32)
        merged[f"{stem}_+"] = 0.5 * (forward_plus + reverse_minus)
        merged[f"{stem}_-"] = 0.5 * (forward_minus + reverse_plus)

    return np.stack([merged[key] for key in canonical_keys], axis=0)


def infer_token_classification_tracks_with_rc(
    sequence: str,
    tokenizer,
    model,
    label_names: Sequence[str],
    window_size: int,
    context_fraction: float,
    average_token_length: float,
    max_genomic_chunk_ratio: float,
    drop_last: bool,
    gap_token_id: Optional[int],
    batch_size: int,
    device: torch.device,
    apply_sigmoid: bool,
    rc_task: str,
    use_reverse_complement: bool,
    progress_desc: str,
    logger: Optional[logging.Logger] = None,
    chunk_log_every: int = 1000,
) -> np.ndarray:
    del batch_size
    del rc_task
    logger = logger or LOGGER

    forward_tracks = _infer_exporter_style_tracks_single_strand(
        sequence=sequence,
        tokenizer=tokenizer,
        model=model,
        label_names=label_names,
        model_input_length_token=int(window_size),
        average_token_length=float(average_token_length),
        chunk_overlap=float(context_fraction),
        max_genomic_chunk_ratio=float(max_genomic_chunk_ratio),
        drop_last=bool(drop_last),
        gap_token_id=gap_token_id,
        device=device,
        apply_sigmoid=apply_sigmoid,
        reference_strand="+",
        progress_desc=f"{progress_desc} [forward]",
        logger=logger,
        chunk_log_every=int(chunk_log_every),
    )
    if not use_reverse_complement:
        merged = _merge_forward_and_rc_track_dicts(label_names, forward_tracks, None)
        logger.info("%s | RC disabled | merged_shape=%s", progress_desc, tuple(merged.shape))
        return merged

    reverse_tracks = _infer_exporter_style_tracks_single_strand(
        sequence=sequence,
        tokenizer=tokenizer,
        model=model,
        label_names=label_names,
        model_input_length_token=int(window_size),
        average_token_length=float(average_token_length),
        chunk_overlap=float(context_fraction),
        max_genomic_chunk_ratio=float(max_genomic_chunk_ratio),
        drop_last=bool(drop_last),
        gap_token_id=gap_token_id,
        device=device,
        apply_sigmoid=apply_sigmoid,
        reference_strand="-",
        progress_desc=f"{progress_desc} [RC]",
        logger=logger,
        chunk_log_every=int(chunk_log_every),
    )
    merged = _merge_forward_and_rc_track_dicts(label_names, forward_tracks, reverse_tracks)
    logger.info("%s | merged forward+RC shape=%s", progress_desc, tuple(merged.shape))
    return merged


def infer_overlap_tracks_single_orientation(
    sequence: str,
    tokenizer,
    model,
    label_names: Sequence[str],
    chunk_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    apply_sigmoid: bool,
    progress_desc: str,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    del batch_size
    logger = logger or LOGGER
    seq_length = len(sequence)
    num_labels = len(label_names)
    plan = build_overlap_window_plan(seq_length, chunk_size, overlap)
    aggregate = np.zeros((num_labels, seq_length), dtype=np.float32)
    counts = np.zeros(seq_length, dtype=np.float32)
    use_fast = bool(getattr(tokenizer, "is_fast", False))

    with torch.no_grad():
        for start, end in tqdm(plan, desc=progress_desc, total=len(plan), leave=False):
            text = sequence[start:end]
            if not text:
                continue

            return_token_type_ids = "token_type_ids" in getattr(tokenizer, "model_input_names", [])
            tokenizer_kwargs = {
                "add_special_tokens": True,
                "return_attention_mask": True,
                "return_token_type_ids": return_token_type_ids,
                "return_special_tokens_mask": True,
                "return_tensors": "pt",
                "truncation": False,
                "verbose": False,
            }
            if use_fast:
                tokenizer_kwargs["return_offsets_mapping"] = True

            encoding = tokenizer(text, **tokenizer_kwargs)
            special_mask = _extract_special_mask_from_encoding(encoding, tokenizer)
            offsets = None
            if use_fast:
                offsets = _offsets_to_numpy(encoding.pop("offset_mapping"))
                if offsets.ndim > 2:
                    offsets = offsets[0]

            model_inputs = _select_model_inputs_from_encoding(encoding, model, device)
            outputs = model(**model_inputs)
            scores = _extract_prediction_tensor(outputs, preferred_key="predicts").detach().float()[0]
            if apply_sigmoid:
                scores = scores.sigmoid()
            score_np = scores.cpu().numpy()

            token_scores = score_np[~special_mask]
            window_len = end - start
            if use_fast and offsets is not None:
                window_offsets = offsets[~special_mask]
                if token_scores.shape[0] != len(window_offsets):
                    usable = min(token_scores.shape[0], len(window_offsets))
                    logger.warning(
                        "Token/logit length mismatch (%d logits vs %d offsets); clipping to %d.",
                        token_scores.shape[0],
                        len(window_offsets),
                        usable,
                    )
                    token_scores = token_scores[:usable]
                    window_offsets = window_offsets[:usable]
                window_tracks = token_scores_to_base_tracks(
                    token_scores=token_scores,
                    offsets=window_offsets,
                    sequence_length=window_len,
                    num_labels=num_labels,
                )
            else:
                window_tracks = _project_token_scores_without_offsets(
                    token_scores=token_scores,
                    sequence_length=window_len,
                    num_labels=num_labels,
                )

            aggregate[:, start:end] += window_tracks[:, :window_len]
            counts[start:end] += 1.0

    covered = counts > 0
    if np.any(covered):
        aggregate[:, covered] /= counts[covered][None, :]
    return aggregate


def infer_segmentation_tracks_with_rc(
    sequence: str,
    tokenizer,
    model,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    use_reverse_complement: bool,
    logger: Optional[logging.Logger] = None,
    progress_desc: str = "Segmentation",
    apply_sigmoid: bool = True,
) -> np.ndarray:
    logger = logger or LOGGER
    label_names = resolve_label_names(model.config, SEGMENTATION_LABELS_DEFAULT)
    forward = infer_overlap_tracks_single_orientation(
        sequence=sequence,
        tokenizer=tokenizer,
        model=model,
        label_names=label_names,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        apply_sigmoid=apply_sigmoid,
        progress_desc=progress_desc,
        logger=logger,
    )
    if not use_reverse_complement:
        return forward

    reverse = infer_overlap_tracks_single_orientation(
        sequence=reverse_complement(sequence),
        tokenizer=tokenizer,
        model=model,
        label_names=label_names,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
        apply_sigmoid=apply_sigmoid,
        progress_desc=f"{progress_desc} [RC]",
        logger=logger,
    )
    permutation = build_channel_permutation(label_names, "segmentation")
    reverse_mapped = reverse[permutation, ::-1]
    return 0.5 * (forward + reverse_mapped)


def infer_sequence_classification_score_with_rc(
    sequence: str,
    tokenizer,
    model,
    chunk_size: int,
    overlap: int,
    batch_size: int,
    device: torch.device,
    use_reverse_complement: bool,
    logger: Optional[logging.Logger] = None,
    apply_sigmoid: bool = True,
) -> float:
    del overlap
    del batch_size
    logger = logger or LOGGER
    accepted_args = set(inspect.signature(model.forward).parameters)

    def infer_one(seq: str, desc: str) -> float:
        logger.debug("%s on %d nt", desc, len(seq))
        encoding = tokenizer(
            seq,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=int(chunk_size),
            verbose=False,
        )
        model_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in encoding.items()
            if key in accepted_args
        }
        with torch.no_grad():
            outputs = model(**model_inputs)
        logits = _extract_prediction_tensor(outputs, preferred_key="predicts").detach().float()
        if logits.ndim != 2 or logits.shape[0] != 1:
            raise ValueError(f"Unsupported transcript-type logits shape: {tuple(logits.shape)}")
        if logits.shape[1] == 1:
            value = logits[0, 0]
            if apply_sigmoid:
                value = torch.sigmoid(value)
            return float(value.item())
        probs = torch.softmax(logits, dim=-1)
        if probs.shape[1] < 2:
            return float(probs[0, 0].item())
        return float(probs[0, 1].item())

    forward_prob = infer_one(sequence, "Transcript type")
    if not use_reverse_complement:
        return forward_prob

    reverse_prob = infer_one(reverse_complement(sequence), "Transcript type [RC]")
    return 0.5 * (forward_prob + reverse_prob)

########################################
# Boundary discovery post-processing
########################################

def merge_rev_comp(signal_plus: np.ndarray, signal_minus: np.ndarray) -> np.ndarray:
    return np.mean([signal_plus, signal_minus], axis=0)


def fft_lowpass(x: np.ndarray, frac: float) -> np.ndarray:
    Xf = np.fft.rfft(x)
    k = int(np.clip(frac, 0.0, 1.0) * len(Xf))
    if k < 1:
        k = 1
    X_lp = np.zeros_like(Xf)
    X_lp[:k] = Xf[:k]
    y = np.fft.irfft(X_lp, n=len(x))
    return y


def call_peaks_on_segment(
    x: np.ndarray,
    lp_frac: float,
    pk_prom: float,
    pk_dist: int,
    pk_height: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    y_lp = fft_lowpass(x, frac=lp_frac)
    idx, _props = find_peaks(y_lp, prominence=pk_prom, distance=pk_dist, height=pk_height)
    return idx, y_lp


def peak_finding(X: np.ndarray, LP_FRAC: float, PK_PROM: float, PK_DIST: int, PK_HEIGHT: Optional[float], logger: Optional[logging.Logger] = None) -> np.ndarray:
    logger = logger or LOGGER
    X = np.nan_to_num(X)
    N = X.shape[1]
    logger.info(
        "Peak finding | input_shape=%s | lp_frac=%.4f | pk_prom=%.4f | pk_dist=%d | pk_height=%s",
        tuple(X.shape),
        float(LP_FRAC),
        float(PK_PROM),
        int(PK_DIST),
        str(PK_HEIGHT),
    )
    mask = np.zeros((4, N), dtype=np.uint8)
    channel_names = ["TSS+", "PolyA+", "TSS-", "PolyA-"]
    for r in range(4):
        x = X[r, :].astype(float, copy=False)
        idx_local, y_lp = call_peaks_on_segment(
            x,
            lp_frac=LP_FRAC,
            pk_prom=PK_PROM,
            pk_dist=PK_DIST,
            pk_height=PK_HEIGHT,
        )
        mask[r, idx_local] = 1
        preview = idx_local[:10].tolist()
        logger.info(
            "Peak finding | channel=%s | raw_max=%.5f | smooth_max=%.5f | peaks=%d | first10=%s",
            channel_names[r],
            float(np.max(x)) if x.size else 0.0,
            float(np.max(y_lp)) if y_lp.size else 0.0,
            len(idx_local),
            preview,
        )
    return mask


def find_tss_polya_pairs_right_left_only(
    arr: np.ndarray,
    chrom_name: str,
    window_size: int = 2_000_000,
    k: int = 10,
    progress_every: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> list[TranscriptInterval]:
    logger = logger or LOGGER
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError("arr must have shape (4, X) in order: TSS+, PolyA+, TSS-, PolyA-")
    X = int(arr.shape[1])
    if window_size > X:
        window_size = X

    tss_plus_idx = np.flatnonzero(arr[0].astype(np.bool_, copy=False))
    polya_plus_idx = np.flatnonzero(arr[1].astype(np.bool_, copy=False))
    tss_minus_idx = np.flatnonzero(arr[2].astype(np.bool_, copy=False))
    polya_minus_idx = np.flatnonzero(arr[3].astype(np.bool_, copy=False))

    tss_plus_idx.sort()
    polya_plus_idx.sort()
    tss_minus_idx.sort()
    polya_minus_idx.sort()

    logger.info(
        "Pairing | peaks counts | TSS+=%d PolyA+=%d TSS-=%d PolyA-=%d | window_size=%d | k=%s",
        len(tss_plus_idx),
        len(polya_plus_idx),
        len(tss_minus_idx),
        len(polya_minus_idx),
        int(window_size),
        str(k),
    )

    pairs_sign: dict[tuple[int, int], str] = {}

    def choose_k_nearest(seed: int, candidates: np.ndarray) -> np.ndarray:
        if k is None or k <= 0 or candidates.size <= k:
            return candidates
        dist = np.abs(candidates - seed)
        idx = np.argpartition(dist, k - 1)[:k]
        order = np.lexsort((candidates[idx], dist[idx]))
        return candidates[idx][order]

    def scan_tss_to_polya_one_sided(
        seeds_tss: np.ndarray,
        targets_polya: np.ndarray,
        direction: str,
        strand_sign: str,
        label: str,
    ) -> None:
        if progress_every:
            logger.info("Scanning %s TSS seeds on strand %s", len(seeds_tss), label)
        for ii, i in enumerate(seeds_tss):
            i = int(i)
            if direction == "right":
                start_w = i
                end_w = min(X, i + window_size)
                left = targets_polya.searchsorted(start_w, side="left")
                right = targets_polya.searchsorted(end_w, side="left")
            elif direction == "left":
                start_w = max(0, i - window_size)
                end_w = i + 1
                left = targets_polya.searchsorted(start_w, side="left")
                right = targets_polya.searchsorted(end_w, side="left")
            else:
                raise ValueError("direction must be 'right' or 'left'")

            if right > left:
                js_window = targets_polya[left:right]
                js_use = choose_k_nearest(i, js_window)
                for j in js_use:
                    j = int(j)
                    a, b = (i, j) if i <= j else (j, i)
                    pairs_sign.setdefault((a, b), strand_sign)

            if progress_every and (ii + 1) % progress_every == 0:
                logger.info("processed %d/%d seeds; pairs=%d", ii + 1, len(seeds_tss), len(pairs_sign))

    scan_tss_to_polya_one_sided(tss_plus_idx, polya_plus_idx, "right", "+", "plus")
    scan_tss_to_polya_one_sided(tss_minus_idx, polya_minus_idx, "left", "-", "minus")

    pairs_sorted = sorted(pairs_sign.keys(), key=lambda ab: (ab[0], ab[1]))
    final_pairs = [TranscriptInterval(chrom_name, a, b, pairs_sign[(a, b)], []) for a, b in pairs_sorted]
    logger.info(
        "Pairing | constructed %d candidate intervals | first10=%s",
        len(final_pairs),
        [(iv.start, iv.end, iv.strand) for iv in final_pairs[:10]],
    )
    return final_pairs


def filter_intervals_by_intragenic(
    pairs: Sequence[TranscriptInterval],
    intragenic_plus: np.ndarray,
    intragenic_minus: np.ndarray,
    prob_threshold: float,
    zero_fraction_drop_threshold: float,
    logger: Optional[logging.Logger] = None,
) -> list[TranscriptInterval]:
    logger = logger or LOGGER
    kept: list[TranscriptInterval] = []
    inspected_preview: list[tuple[int, int, str, float, float]] = []

    logger.info(
        "Intragenic filtering | pairs=%d | prob_threshold=%.4f | zero_fraction_drop_threshold=%.4f",
        len(pairs),
        float(prob_threshold),
        float(zero_fraction_drop_threshold),
    )
    logger.info(
        "Intragenic tracks summary | plus max=%.5f mean=%.5f | minus max=%.5f mean=%.5f",
        float(np.max(intragenic_plus)) if intragenic_plus.size else 0.0,
        float(np.mean(intragenic_plus)) if intragenic_plus.size else 0.0,
        float(np.max(intragenic_minus)) if intragenic_minus.size else 0.0,
        float(np.mean(intragenic_minus)) if intragenic_minus.size else 0.0,
    )

    for interval in tqdm(pairs, desc="Interval filtering", leave=False):
        if interval.end <= interval.start:
            continue
        if interval.strand == "+":
            signal = intragenic_plus[interval.start : interval.end]
        elif interval.strand == "-":
            signal = intragenic_minus[interval.start : interval.end]
        else:
            signal = np.maximum(
                intragenic_plus[interval.start : interval.end],
                intragenic_minus[interval.start : interval.end],
            )
        if signal.size == 0:
            continue
        binary = (signal > prob_threshold).astype(np.uint8)
        zero_fraction = float((binary == 0).mean())
        signal_mean = float(np.mean(signal))
        if len(inspected_preview) < 10:
            inspected_preview.append((int(interval.start), int(interval.end), interval.strand, signal_mean, zero_fraction))
        if zero_fraction <= zero_fraction_drop_threshold:
            kept.append(interval)

    logger.info("Intragenic filtering preview (first10) | %s", inspected_preview)
    logger.info("Kept %d / %d intervals after intragenic filtering.", len(kept), len(pairs))
    return kept


########################################
# Segmentation helpers
########################################

def mask_to_segments(binary_mask: np.ndarray, new_start: int) -> list[tuple[int, int]]:
    one_idx = np.where(binary_mask == 1)[0]
    if one_idx.size == 0:
        return []
    split_points = np.where(np.diff(one_idx) > 1)[0] + 1
    groups = np.split(one_idx, split_points)
    return [(int(g[0]) + new_start, int(g[-1]) + 1 + new_start) for g in groups]


def max_predicted_mask(prob_tensor: np.ndarray, track: str, is_cds_binary: bool = False) -> np.ndarray:
    labels = ["5UTR", "exon", "intron", "3UTR", "CDS"]
    class_index = {lab: i for i, lab in enumerate(labels)}

    if track == "exon":
        keep = np.array([class_index["exon"], class_index["intron"]], dtype=int)
        chosen = prob_tensor[keep, :]
        max_values = np.max(chosen, axis=0)
        return (chosen[0] == max_values).astype(np.uint8)

    if track == "CDS":
        if is_cds_binary:
            cds = prob_tensor[class_index["CDS"], :]
            return (cds >= 0.5).astype(np.uint8)
        keep = np.array(
            [class_index["CDS"], class_index["intron"], class_index["5UTR"], class_index["3UTR"]],
            dtype=int,
        )
        chosen = prob_tensor[keep, :]
        max_values = np.max(chosen, axis=0)
        return (chosen[0] == max_values).astype(np.uint8)

    raise ValueError(f"Unsupported track={track}")


def reverse_complements(kmers: Iterable[str]) -> list[str]:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return ["".join(comp.get(c, "N") for c in s[::-1]) for s in kmers]


def _rc2(two: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
    if len(two) != 2:
        return "NN"
    return comp.get(two[1], "N") + comp.get(two[0], "N")


def _has_boundary_motif(seq: str, interval_start: int, pos_abs: int, role: str, strand: str) -> bool:
    L = len(seq)
    off = pos_abs - interval_start
    if strand == "+":
        if role == "first_end":
            return (0 <= off <= L - 2) and (seq[off : off + 2] in RIGHT_SET)
        return (2 <= off <= L) and (seq[off - 2 : off] in LEFT_SET)

    if role == "first_end":
        if not (0 <= off <= L - 2):
            return False
        return _rc2(seq[off : off + 2]) in LEFT_SET
    if not (2 <= off <= L):
        return False
    return _rc2(seq[off - 2 : off]) in RIGHT_SET


def _nearest_motif_unbounded(
    seq: str,
    interval_start: int,
    current_abs: int,
    role: str,
    strand: str,
    lo_abs: int,
    hi_abs: int,
) -> Optional[int]:
    L = len(seq)
    if role == "first_end":
        lo_abs = max(lo_abs, interval_start)
        hi_abs = min(hi_abs, interval_start + L - 2)
    else:
        lo_abs = max(lo_abs, interval_start + 2)
        hi_abs = min(hi_abs, interval_start + L)
    if lo_abs > hi_abs:
        return None

    d = 0
    while True:
        left = current_abs - d
        right = current_abs + d
        if lo_abs <= left <= hi_abs and _has_boundary_motif(seq, interval_start, left, role, strand):
            return left
        if d != 0 and lo_abs <= right <= hi_abs and _has_boundary_motif(seq, interval_start, right, role, strand):
            return right
        if left <= lo_abs and right >= hi_abs:
            return None
        d += 1


def _check_splice_v2_style_internal(
    seq: str,
    exons: list[tuple[int, int]],
    pred_start: int,
    pred_strand: str,
) -> set[tuple[int, int]]:
    if not exons or seq is None:
        return set()
    noshort = [(s, e) for (s, e) in exons if (e - s) >= 3]
    if not noshort:
        return set()
    noshort.sort(key=lambda x: x[0])
    n = len(noshort)
    if n < 3:
        return set(noshort)

    if pred_strand == "+":
        left_list = LEFT_SET
        right_list = RIGHT_SET
    else:
        left_list = LEFT_SET
        right_list = {_rc2(x) for x in LEFT_SET}

    kept = {noshort[0], noshort[-1]}
    L = len(seq)
    for i in range(1, n - 1):
        s, e = noshort[i]
        i0 = s - pred_start
        j0 = e - pred_start
        ok_left = i0 >= 2 and i0 <= L and seq[i0 - 2 : i0] in left_list
        ok_right = j0 >= 0 and j0 + 2 <= L and seq[j0 : j0 + 2] in right_list
        if ok_left and ok_right:
            kept.add((s, e))
    return kept


def apply_splice_filter_for_exon(
    seq: str,
    exons: list[tuple[int, int]],
    interval_start: int,
    strand: str,
) -> list[tuple[int, int]]:
    if not exons:
        return []
    exons_sorted = sorted(exons, key=lambda x: x[0])
    kept_set = _check_splice_v2_style_internal(seq, exons_sorted, interval_start, strand)
    kept = [seg for seg in exons_sorted if seg in kept_set]
    if len(kept) <= 1:
        return kept

    first_s, first_e = kept[0]
    last_s, last_e = kept[-1]

    if not _has_boundary_motif(seq, interval_start, first_e, role="first_end", strand=strand):
        next_start = kept[1][0]
        new_e = _nearest_motif_unbounded(seq, interval_start, first_e, "first_end", strand, lo_abs=first_s + 1, hi_abs=next_start)
        if new_e is not None and new_e > first_s and new_e <= next_start:
            first_e = new_e

    if not _has_boundary_motif(seq, interval_start, last_s, role="last_start", strand=strand):
        prev_end = kept[-2][1]
        new_s = _nearest_motif_unbounded(seq, interval_start, last_s, "last_start", strand, lo_abs=prev_end, hi_abs=last_e - 1)
        if new_s is not None and new_s >= prev_end and new_s < last_e:
            last_s = new_s

    kept[0] = (first_s, first_e)
    kept[-1] = (last_s, last_e)

    cleaned = []
    prev_end = None
    for s, e in kept:
        if e <= s:
            continue
        if prev_end is not None and s < prev_end:
            s = prev_end
            if e <= s:
                continue
        cleaned.append((s, e))
        prev_end = e
    return cleaned


def apply_splice_filter_for_cds(
    seq: str,
    cds: list[tuple[int, int]],
    interval_start: int,
    strand: str,
    labels_argmax: np.ndarray,
) -> list[tuple[int, int]]:
    if not cds:
        return []
    cds_sorted = sorted(cds, key=lambda x: x[0])
    kept_set = _check_splice_v2_style_internal(seq, cds_sorted, interval_start, strand)
    kept = [seg for seg in cds_sorted if seg in kept_set]
    if len(kept) <= 1:
        return kept

    first_s, first_e = kept[0]
    last_s, last_e = kept[-1]

    pos_rel_end = first_e - interval_start
    right_is_intron = bool(0 <= pos_rel_end < len(labels_argmax) and labels_argmax[pos_rel_end] == INTRON_IDX)
    if not _has_boundary_motif(seq, interval_start, first_e, role="first_end", strand=strand) and right_is_intron:
        next_start = kept[1][0]
        new_e = _nearest_motif_unbounded(seq, interval_start, first_e, "first_end", strand, lo_abs=first_s + 1, hi_abs=next_start)
        if new_e is not None and new_e > first_s and new_e <= next_start:
            first_e = new_e

    pos_rel_left = (last_s - 1) - interval_start
    left_is_intron = bool(0 <= pos_rel_left < len(labels_argmax) and labels_argmax[pos_rel_left] == INTRON_IDX)
    if not _has_boundary_motif(seq, interval_start, last_s, role="last_start", strand=strand) and left_is_intron:
        prev_end = kept[-2][1]
        new_s = _nearest_motif_unbounded(seq, interval_start, last_s, "last_start", strand, lo_abs=prev_end, hi_abs=last_e - 1)
        if new_s is not None and new_s >= prev_end and new_s < last_e:
            last_s = new_s

    kept[0] = (first_s, first_e)
    kept[-1] = (last_s, last_e)

    cleaned = []
    prev_end = None
    for s, e in kept:
        if e <= s:
            continue
        if prev_end is not None and s < prev_end:
            s = prev_end
            if e <= s:
                continue
        cleaned.append((s, e))
        prev_end = e
    return cleaned


def build_introns_from_exons(exons: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(exons) < 2:
        return []
    return [(exons[i][1], exons[i + 1][0]) for i in range(len(exons) - 1)]


def build_segmentation_result(
    interval_start: int,
    sequence: str,
    strand: str,
    segmentation_tracks: np.ndarray,
    splice_filter: bool,
    is_cds_binary: bool = False,
) -> SegmentationResult:
    exon_mask = max_predicted_mask(segmentation_tracks, "exon")
    cds_mask = max_predicted_mask(segmentation_tracks, "CDS", is_cds_binary=is_cds_binary)
    labels_argmax = np.argmax(segmentation_tracks, axis=0).astype(np.int8)

    exon_segments = mask_to_segments(exon_mask, new_start=interval_start)
    cds_segments = mask_to_segments(cds_mask, new_start=interval_start)

    if splice_filter:
        exon_segments = apply_splice_filter_for_exon(
            seq=sequence,
            exons=exon_segments,
            interval_start=interval_start,
            strand=strand,
        )
        cds_segments = apply_splice_filter_for_cds(
            seq=sequence,
            cds=cds_segments,
            interval_start=interval_start,
            strand=strand,
            labels_argmax=labels_argmax,
        )

    return SegmentationResult(
        exon_segments=exon_segments,
        cds_segments=cds_segments,
        labels_argmax=labels_argmax,
        tracks=segmentation_tracks,
    )
