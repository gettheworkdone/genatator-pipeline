---
library_name: transformers
tags:
  - genomics
  - gene-annotation
  - dna-language-models
  - biology
  - ab-initio
---

# GENATATOR-PIPELINE

GENATATOR-PIPELINE is a Hugging Face pipeline for **ab initio gene annotation from genomic DNA**. The pipeline accepts a FASTA file, detects transcript intervals, assigns transcript class, resolves exon-intron structure, and writes the final annotation as a GFF file in GFF3 format.

The pipeline combines interval discovery, transcript-type classification, segmentation, and GFF generation in a single Hugging Face pipeline call. Input is a FASTA file, and output is a single string containing the path to the written GFF file.

## Hugging Face pipeline usage

### Basic example

```python
from transformers import pipeline

pipe = pipeline(
    task="genatator-pipeline",
    model="shmelev/genatator-pipeline",
    trust_remote_code=True,
    device=0,
)

output_path = pipe(
    "genome.fasta",
    output_gff_path="genome.gff",
)
print(output_path)
````

### Example with all supported parameters defined

```python
from transformers import pipeline

pipe = pipeline(
    task="genatator-pipeline",
    model="shmelev/genatator-pipeline",
    trust_remote_code=True,
    device=0,
    dtype="float32",
    edge_model_path="shmelev/genatator-moderngena-base-multispecies-edge-model",
    region_model_path="shmelev/genatator-moderngena-base-multispecies-region-model",
    transcript_type_model_path="shmelev/genatator-caduceus-ps-multispecies-transcript-type",
    segmentation_model_path="shmelev/genatator-caduceus-ps-multispecies-segmentation",
    edge_context_length=1024,
    region_context_length=8192,
    transcript_type_context_length=250000,
    segmentation_context_length=250000,
    edge_average_token_length=9.0,
    region_average_token_length=9.0,
    edge_max_genomic_chunk_ratio=1.5,
    region_max_genomic_chunk_ratio=1.5,
    edge_drop_last=False,
    region_drop_last=False,
    edge_apply_sigmoid=False,
    region_apply_sigmoid=False,
    transcript_type_apply_sigmoid=True,
    segmentation_apply_sigmoid=True,
    edge_gap_token_id=5,
    region_gap_token_id=5,
)

output_path = pipe(
    "genome.fasta",
    output_gff_path="genome.gff",
    edge_context_fraction=0.5,
    region_context_fraction=0.5,
    gene_finding_use_reverse_complement=True,
    transcript_type_use_reverse_complement=True,
    segmentation_use_reverse_complement=True,
    lp_frac=0.05,
    pk_prom=0.1,
    pk_dist=50,
    pk_height=None,
    interval_window_size=2_000_000,
    max_pairs_per_seed=10,
    prob_threshold=0.5,
    zero_fraction_drop_threshold=0.01,
    transcript_type_threshold=0.5,
    splice_filter=True,
    deduplicate=True,
    intronic_filtering=False,
    keep_longest_terminal_variant=False,
    use_cds_heuristic=True,
    save_intermediate_files=False,
    intermediate_output_dir=None,
    pairing_progress_every=1000,
    chunk_log_every=1000,
    shift=None,
)
print(output_path)
```

All four stage models run with **batch size 1**.

## Parameter reference

### Model repositories

* `edge_model_path` — Hugging Face repository of the edge model that predicts `TSS+`, `TSS-`, `PolyA+`, and `PolyA-` signals.
* `region_model_path` — Hugging Face repository of the region model that provides the intragenic strand-specific tracks used for interval filtering.
* `transcript_type_model_path` — Hugging Face repository of the interval-level classifier that assigns `mRNA` or `lnc_RNA`.
* `segmentation_model_path` — Hugging Face repository of the segmentation model that predicts transcript structure inside each interval.

### Context-length parameters

* `edge_context_length` — Token length of each edge-model input, including the system tokens added by the tokenizer.
* `region_context_length` — Token length of each region-model input, including the system tokens added by the tokenizer. The default is 8192 tokens, matching the gene-finding benchmark and manuscript configuration.
* `transcript_type_context_length` — Total token length passed to the transcript-type model, including system tokens. Only the leading prefix up to this context is processed.
* `segmentation_context_length` — Nucleotide length of each segmentation-model inference block. Segmentation uses consecutive non-overlapping blocks.
* `edge_average_token_length` — Average BPE token length, in nucleotides, used to convert the token context length of the edge model into the genomic chunk length before tokenization.
* `region_average_token_length` — Average BPE token length, in nucleotides, used to convert the token context length of the region model into the genomic chunk length before tokenization.
* `edge_max_genomic_chunk_ratio` — Ratio between the extended genomic extraction length and the nominal edge-model genomic chunk length before tokenizer truncation.
* `region_max_genomic_chunk_ratio` — Ratio between the extended genomic extraction length and the nominal region-model genomic chunk length before tokenizer truncation.
* `edge_drop_last` — If `True`, follows the reference chunk builder and omits the final incomplete edge-model chunk from the sequential chunk index construction. The default is `False`.
* `region_drop_last` — If `True`, follows the reference chunk builder and omits the final incomplete region-model chunk from the sequential chunk index construction. The default is `False`.
* `edge_gap_token_id` — Gap-token identifier used to correct edge-model offset mappings in the same way as the reference inference dataset.
* `region_gap_token_id` — Gap-token identifier used to correct region-model offset mappings in the same way as the reference inference dataset.

### Interval-discovery parameters

* `edge_context_fraction` — Genomic overlap fraction used when constructing sequential edge-model genomic chunks before tokenization.
* `region_context_fraction` — Genomic overlap fraction used when constructing sequential region-model genomic chunks before tokenization.
* `lp_frac` — Fraction of the Fourier spectrum retained by the low-pass smoother before peak detection.
* `pk_prom` — Minimum peak prominence used during boundary detection.
* `pk_dist` — Minimum distance between neighbouring peaks.
* `pk_height` — Optional minimum height threshold for peaks after smoothing. Use `None` to disable this constraint.
* `interval_window_size` — Maximum genomic span allowed when pairing a TSS peak with a PolyA peak on the same strand.
* `max_pairs_per_seed` — Maximum number of nearest PolyA peaks retained for each strand-consistent TSS seed during interval pairing.
* `prob_threshold` — Intragenic probability threshold used to binarize the region-model signal inside each candidate interval.
* `zero_fraction_drop_threshold` — Maximum tolerated fraction of bases below `prob_threshold` inside a candidate interval.

### Reverse-complement options

* `gene_finding_use_reverse_complement` — Enables forward/reverse-complement averaging for the edge and region models.
* `transcript_type_use_reverse_complement` — Enables forward/reverse-complement averaging for transcript-type classification.
* `segmentation_use_reverse_complement` — Enables forward/reverse-complement averaging for segmentation.

### Activation options

* `edge_apply_sigmoid` — Applies an additional sigmoid to the edge-model output channels before token-to-nucleotide projection. The default is `False`.
* `region_apply_sigmoid` — Applies an additional sigmoid to the region-model output channels before token-to-nucleotide projection. The default is `False`.
* `transcript_type_apply_sigmoid` — Applies a sigmoid to single-logit transcript-type outputs before thresholding. The default is `True`. For multi-logit transcript-type outputs, the pipeline continues to use softmax.
* `segmentation_apply_sigmoid` — Applies an additional sigmoid to the segmentation-model output channels before structural decoding. The default is `True`.

### Transcript and segmentation parameters

* `transcript_type_threshold` — Threshold applied to the predicted `lnc_RNA` probability. Intervals at or above this value are labeled `lnc_RNA`; intervals below it are labeled `mRNA`.
* `splice_filter` — Enables splice-motif filtering and terminal splice-boundary correction for exon and CDS segments.
* `deduplicate` — Removes duplicate transcript predictions before GFF export. Duplicates are transcripts with identical interval boundaries and identical internal annotation structure (exons, introns, CDS). Default: `True`.
* `intronic_filtering` — Drops an interval prediction when the first or the last nucleotide of the final segmentation argmax is classified as `intron`. This check is applied after forward/RC averaging (if enabled). Default: `False`.
* `keep_longest_terminal_variant` — For transcripts grouped under the same gene, keeps only the longest transcript among variants that share the same internal exon structure and differ only by first-exon start and/or last-exon end. Default: `False`.
* `use_cds_heuristic` — Replaces the predicted CDS with the exon-derived CDS heuristic used in the accompanying benchmark code. This option affects `mRNA` transcripts only.

### General inference parameters

* `save_intermediate_files` — If `True`, writes gene-finding intermediate artifacts for each FASTA record, including `.npy` tracks, `.bed` interval files, and a compressed `.h5` debug dump when `h5py` is installed.
* `intermediate_output_dir` — Optional output directory for intermediate artifacts. If omitted, intermediate files are written next to the input FASTA file.
* `pairing_progress_every` — Logging period, in TSS seeds, used during candidate interval construction.
* `chunk_log_every` — Logging period, in genomic chunks, used during edge and region inference.
* `shift` — Coordinate offset applied to the final annotation. This may be an integer or the string `"UCSC"`. In `"UCSC"` mode, coordinates are shifted according to FASTA headers of the form `chrom:start-end`.
* `output_gff_path` — Path to the GFF file written by the pipeline call.

Standard `transformers.pipeline(...)` arguments such as `device` and `dtype` are also supported.

## What the pipeline does

### 1) Interval discovery

The first stage identifies transcript intervals with two strand-aware DNA language models:

* `edge_model_path` detects transcription start sites (TSS) and polyadenylation sites (PolyA)
* `region_model_path` provides intragenic signal used to filter candidate intervals

For these two models, the genomic sequence is processed with the same chunking logic as the reference chromosome-evaluation pipeline. Each stage first builds overlapping genomic windows in nucleotide space using the requested token context length, an average token length estimate, and a maximum genomic expansion ratio. Every genomic window is then tokenized with `truncation=True`, `padding='max_length'`, and `return_offsets_mapping=True`. The first and last system-token positions are discarded, token scores are projected back to genomic coordinates with the tokenizer offsets, and overlapping windows are averaged in the chromosome-length tracks. When reverse-complement averaging is enabled, forward predictions are merged with the strand-swapped reverse-complement tracks exactly as in the reference exporter logic before peak calling and intragenic filtering.

### 2) Transcript-type assignment

Each retained interval is classified by `transcript_type_model_path` as either:

* `mRNA`
* `lnc_RNA`

Only the leading token prefix defined by `transcript_type_context_length` is evaluated. Sequence beyond this context is not processed. When reverse-complement averaging is enabled for this stage, the forward and reverse-complement predictions are averaged.

### 3) Segmentation

Each interval is segmented by `segmentation_model_path` into nucleotide-level structural classes. Exons are derived from the exon-versus-intron competition, and CDS segments are derived from the CDS-versus-non-CDS competition. Segmentation is stitched from **non-overlapping** interval blocks. When the segmentation tokenizer provides offset mappings, token-level outputs are projected exactly to nucleotide coordinates; otherwise the pipeline uses the tokenizer-specific non-fast branch implemented for the Caduceus segmentation stage.

### 4) GFF generation

The final annotation contains:

* `gene`
* `mRNA` or `lnc_RNA`
* `exon`
* `intron`
* `CDS` for `mRNA` transcripts only

No CDS is emitted for `lnc_RNA` transcripts.

## Default model repositories

* `edge_model_path`: `shmelev/genatator-moderngena-base-multispecies-edge-model`
* `region_model_path`: `shmelev/genatator-moderngena-base-multispecies-region-model`
* `transcript_type_model_path`: `shmelev/genatator-caduceus-ps-multispecies-transcript-type`
* `segmentation_model_path`: `shmelev/genatator-caduceus-ps-multispecies-segmentation`

## Input and output

**Input**

* Path to a FASTA file
* The FASTA file may contain one record or multiple records

**Output**

* A single Python string: the path to the written `.gff` file
* The file contents follow the GFF3 specification

## Dependencies

Create the Conda environment from `environment.yml` before running the pipeline locally.

```bash
conda env create -f environment.yml
conda activate genatator_pipeline
```

## Output annotation

The written GFF file contains one `gene` feature for each predicted gene locus and one transcript feature for each predicted transcript. Exons and introns are derived from the segmentation stage. CDS features are emitted only for transcripts classified as `mRNA`. The attribute field of each transcript includes `lncRNA_probability`, which stores the score produced by the transcript-type model.

### How transcripts are assigned to the same gene

Transcripts are grouped into the same gene if all of the following are true:

* same chromosome (`chrom`)
* same strand (`+` / `-`)
* same transcript type (`mRNA` or `lnc_RNA`)
* genomic overlap with the current gene span (`transcript.start <= current_gene.end`)

If any of these checks fails, a new gene is started.

## Docker deployment

All Docker assets are stored in `docker/`.

### Build image

```bash
docker build -f docker/Dockerfile -t genatator-pipeline:latest .
```

### Run container

```bash
docker run --gpus all --rm -p 3000:3000 -v "$(pwd)":/data genatator-pipeline:latest
```

### Call Flask API

The container exposes `POST /api/genatator-pipeline/upload` (and keeps `POST /run` for backward compatibility) and expects:

* multipart field `file` — input FASTA file (optional if `dna` is provided)
* form field `dna` — FASTA text or plain DNA sequence text (optional if `file` is provided)

The server writes output files into `/generated/genatator-pipeline/` inside the container and returns a generated path.

Example:

```bash
curl -X POST "http://localhost:3000/api/genatator-pipeline/upload" \
  -F "file=@/data/genome.fasta"
```

Example with DNA text form:

```bash
curl -X POST "http://localhost:3000/api/genatator-pipeline/upload" \
  -F "dna=>seq1\nATGCGTATGCGT"
```

Response example:

```json
{"gff_file":"/generated/genatator-pipeline/request_2026-04-07_123456.gff"}
```

The container startup script installs dependencies in this strict order inside Conda:

1. CUDA base image/runtime
2. PyTorch CUDA 12.1 wheels (`torch`, `torchvision`, `torchaudio`)
3. `causal-conv1d`
4. `mamba-ssm`
5. `flash-attn`
6. Remaining Python dependencies and local pipeline package
