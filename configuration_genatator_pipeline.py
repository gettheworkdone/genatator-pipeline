from __future__ import annotations

from typing import Any, Optional

try:
    from transformers.configuration_utils import PretrainedConfig
except ImportError:
    from transformers.configuration_utils import PreTrainedConfig as PretrainedConfig


class GenatatorPipelineConfig(PretrainedConfig):
    """Configuration for the GENATATOR Hugging Face pipeline."""

    model_type = "genatator_pipeline"

    def __init__(
        self,
        edge_model_path: str = "shmelev/genatator-moderngena-base-multispecies-edge-model",
        region_model_path: str = "shmelev/genatator-moderngena-base-multispecies-region-model",
        transcript_type_model_path: str = "shmelev/genatator-caduceus-ps-multispecies-transcript-type",
        segmentation_model_path: str = "shmelev/genatator-caduceus-ps-multispecies-segmentation",
        edge_context_length: int = 1024,
        region_context_length: int = 8192,
        transcript_type_context_length: int = 250000,
        segmentation_context_length: int = 250000,
        edge_context_fraction: float = 0.5,
        region_context_fraction: float = 0.5,
        edge_average_token_length: float = 9.0,
        region_average_token_length: float = 9.0,
        edge_max_genomic_chunk_ratio: float = 1.5,
        region_max_genomic_chunk_ratio: float = 1.5,
        edge_drop_last: bool = False,
        region_drop_last: bool = False,
        edge_gap_token_id: int = 5,
        region_gap_token_id: int = 5,
        gene_finding_use_reverse_complement: Optional[bool] = None,
        transcript_type_use_reverse_complement: Optional[bool] = None,
        segmentation_use_reverse_complement: Optional[bool] = None,
        edge_apply_sigmoid: bool = False,
        region_apply_sigmoid: bool = False,
        transcript_type_apply_sigmoid: bool = True,
        segmentation_apply_sigmoid: bool = True,
        lp_frac: float = 0.05,
        pk_prom: float = 0.1,
        pk_dist: int = 50,
        pk_height: Optional[float] = None,
        interval_window_size: int = 2_000_000,
        max_pairs_per_seed: int = 10,
        prob_threshold: float = 0.5,
        zero_fraction_drop_threshold: float = 0.01,
        transcript_type_threshold: float = 0.5,
        splice_filter: bool = True,
        transcript_coloring_thresholds: str | list[float] = "auto",
        use_cds_heuristic: bool = True,
        save_intermediate_files: bool = False,
        intermediate_output_dir: Optional[str] = None,
        pairing_progress_every: int = 1000,
        chunk_log_every: int = 1000,
        shift: Optional[str | int] = None,
        torch_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        edge_context_length = kwargs.pop("edge_window_size", edge_context_length)
        region_context_length = kwargs.pop("region_window_size", region_context_length)
        transcript_type_context_length = kwargs.pop(
            "transcript_type_chunk_size", transcript_type_context_length
        )
        segmentation_context_length = kwargs.pop(
            "segmentation_chunk_size", segmentation_context_length
        )

        legacy_use_reverse_complement = kwargs.pop("use_reverse_complement", None)
        if legacy_use_reverse_complement is not None:
            if gene_finding_use_reverse_complement is None:
                gene_finding_use_reverse_complement = bool(legacy_use_reverse_complement)
            if transcript_type_use_reverse_complement is None:
                transcript_type_use_reverse_complement = bool(legacy_use_reverse_complement)
            if segmentation_use_reverse_complement is None:
                segmentation_use_reverse_complement = bool(legacy_use_reverse_complement)

        if gene_finding_use_reverse_complement is None:
            gene_finding_use_reverse_complement = True
        if transcript_type_use_reverse_complement is None:
            transcript_type_use_reverse_complement = True
        if segmentation_use_reverse_complement is None:
            segmentation_use_reverse_complement = True

        for legacy_name in (
            "genome_batch_size",
            "interval_batch_size",
            "transcript_type_chunk_overlap",
            "segmentation_chunk_overlap",
            "k",
            "gene_merge_gap",
            "save_intermediate_files",
            "cds_heuristic_min_orf_nt",
            "cds_heuristic_require_start_codon",
            "cds_heuristic_require_stop_codon",
            "cds_heuristic_allow_partial_orf",
            "edge_av_token_len",
            "region_av_token_len",
        ):
            kwargs.pop(legacy_name, None)

        self.edge_model_path = str(edge_model_path)
        self.region_model_path = str(region_model_path)
        self.transcript_type_model_path = str(transcript_type_model_path)
        self.segmentation_model_path = str(segmentation_model_path)

        self.edge_context_length = int(edge_context_length)
        self.region_context_length = int(region_context_length)
        self.transcript_type_context_length = int(transcript_type_context_length)
        self.segmentation_context_length = int(segmentation_context_length)

        self.edge_context_fraction = float(edge_context_fraction)
        self.region_context_fraction = float(region_context_fraction)
        self.edge_average_token_length = float(edge_average_token_length)
        self.region_average_token_length = float(region_average_token_length)
        self.edge_max_genomic_chunk_ratio = float(edge_max_genomic_chunk_ratio)
        self.region_max_genomic_chunk_ratio = float(region_max_genomic_chunk_ratio)
        self.edge_drop_last = bool(edge_drop_last)
        self.region_drop_last = bool(region_drop_last)
        self.edge_gap_token_id = int(edge_gap_token_id)
        self.region_gap_token_id = int(region_gap_token_id)

        self.gene_finding_use_reverse_complement = bool(gene_finding_use_reverse_complement)
        self.transcript_type_use_reverse_complement = bool(transcript_type_use_reverse_complement)
        self.segmentation_use_reverse_complement = bool(segmentation_use_reverse_complement)
        self.edge_apply_sigmoid = bool(edge_apply_sigmoid)
        self.region_apply_sigmoid = bool(region_apply_sigmoid)
        self.transcript_type_apply_sigmoid = bool(transcript_type_apply_sigmoid)
        self.segmentation_apply_sigmoid = bool(segmentation_apply_sigmoid)

        self.lp_frac = float(lp_frac)
        self.pk_prom = float(pk_prom)
        self.pk_dist = int(pk_dist)
        self.pk_height = pk_height if pk_height is None else float(pk_height)
        self.interval_window_size = int(interval_window_size)
        self.max_pairs_per_seed = int(max_pairs_per_seed)
        self.prob_threshold = float(prob_threshold)
        self.zero_fraction_drop_threshold = float(zero_fraction_drop_threshold)
        self.transcript_type_threshold = float(transcript_type_threshold)
        self.splice_filter = bool(splice_filter)
        self.transcript_coloring_thresholds = transcript_coloring_thresholds
        self.use_cds_heuristic = bool(use_cds_heuristic)
        self.save_intermediate_files = bool(save_intermediate_files)
        self.intermediate_output_dir = None if intermediate_output_dir in {None, ""} else str(intermediate_output_dir)
        self.pairing_progress_every = int(pairing_progress_every)
        self.chunk_log_every = int(chunk_log_every)
        self.shift = shift
        self.torch_dtype = str(torch_dtype)

        self._validate()
        super().__init__(**kwargs)

    def _validate(self) -> None:
        for name, value in (
            ("edge_model_path", self.edge_model_path),
            ("region_model_path", self.region_model_path),
            ("transcript_type_model_path", self.transcript_type_model_path),
            ("segmentation_model_path", self.segmentation_model_path),
        ):
            if not value:
                raise ValueError(f"{name} must be a non-empty string.")

        for name, value in (
            ("edge_context_length", self.edge_context_length),
            ("region_context_length", self.region_context_length),
            ("transcript_type_context_length", self.transcript_type_context_length),
            ("segmentation_context_length", self.segmentation_context_length),
            ("pk_dist", self.pk_dist),
            ("interval_window_size", self.interval_window_size),
            ("max_pairs_per_seed", self.max_pairs_per_seed),
            ("edge_gap_token_id", self.edge_gap_token_id),
            ("region_gap_token_id", self.region_gap_token_id),
            ("pairing_progress_every", self.pairing_progress_every),
            ("chunk_log_every", self.chunk_log_every),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}.")

        for name, value in (
            ("edge_context_fraction", self.edge_context_fraction),
            ("region_context_fraction", self.region_context_fraction),
        ):
            if not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0), got {value}.")

        for name, value in (
            ("edge_average_token_length", self.edge_average_token_length),
            ("region_average_token_length", self.region_average_token_length),
            ("edge_max_genomic_chunk_ratio", self.edge_max_genomic_chunk_ratio),
            ("region_max_genomic_chunk_ratio", self.region_max_genomic_chunk_ratio),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0, got {value}.")

        for name, value in (
            ("prob_threshold", self.prob_threshold),
            ("zero_fraction_drop_threshold", self.zero_fraction_drop_threshold),
            ("transcript_type_threshold", self.transcript_type_threshold),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}.")

    def to_runtime_defaults(self) -> dict[str, Any]:
        return {
            "edge_model_path": self.edge_model_path,
            "region_model_path": self.region_model_path,
            "transcript_type_model_path": self.transcript_type_model_path,
            "segmentation_model_path": self.segmentation_model_path,
            "edge_context_length": self.edge_context_length,
            "region_context_length": self.region_context_length,
            "transcript_type_context_length": self.transcript_type_context_length,
            "segmentation_context_length": self.segmentation_context_length,
            "edge_context_fraction": self.edge_context_fraction,
            "region_context_fraction": self.region_context_fraction,
            "edge_average_token_length": self.edge_average_token_length,
            "region_average_token_length": self.region_average_token_length,
            "edge_max_genomic_chunk_ratio": self.edge_max_genomic_chunk_ratio,
            "region_max_genomic_chunk_ratio": self.region_max_genomic_chunk_ratio,
            "edge_drop_last": self.edge_drop_last,
            "region_drop_last": self.region_drop_last,
            "edge_gap_token_id": self.edge_gap_token_id,
            "region_gap_token_id": self.region_gap_token_id,
            "gene_finding_use_reverse_complement": self.gene_finding_use_reverse_complement,
            "transcript_type_use_reverse_complement": self.transcript_type_use_reverse_complement,
            "segmentation_use_reverse_complement": self.segmentation_use_reverse_complement,
            "edge_apply_sigmoid": self.edge_apply_sigmoid,
            "region_apply_sigmoid": self.region_apply_sigmoid,
            "transcript_type_apply_sigmoid": self.transcript_type_apply_sigmoid,
            "segmentation_apply_sigmoid": self.segmentation_apply_sigmoid,
            "lp_frac": self.lp_frac,
            "pk_prom": self.pk_prom,
            "pk_dist": self.pk_dist,
            "pk_height": self.pk_height,
            "interval_window_size": self.interval_window_size,
            "max_pairs_per_seed": self.max_pairs_per_seed,
            "prob_threshold": self.prob_threshold,
            "zero_fraction_drop_threshold": self.zero_fraction_drop_threshold,
            "transcript_type_threshold": self.transcript_type_threshold,
            "splice_filter": self.splice_filter,
            "transcript_coloring_thresholds": self.transcript_coloring_thresholds,
            "use_cds_heuristic": self.use_cds_heuristic,
            "save_intermediate_files": self.save_intermediate_files,
            "intermediate_output_dir": self.intermediate_output_dir,
            "pairing_progress_every": self.pairing_progress_every,
            "chunk_log_every": self.chunk_log_every,
            "shift": self.shift,
            "torch_dtype": self.torch_dtype,
        }
