
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from transformers import AutoConfig, PreTrainedModel
except ImportError:
    from transformers import AutoConfig
    from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

try:
    from .configuration_genatator_pipeline import GenatatorPipelineConfig
except ImportError:
    from configuration_genatator_pipeline import GenatatorPipelineConfig


class GenatatorPipelineModel(PreTrainedModel):
    """Minimal wrapper model used only to satisfy Hugging Face pipeline loading.

    The actual biological inference is performed by the four stage models loaded
    inside ``GenatatorPipeline``. This wrapper intentionally carries no weights.
    """

    config_class = GenatatorPipelineConfig
    base_model_prefix = "genatator_pipeline"
    main_input_name = "input_ids"

    def __init__(self, config: GenatatorPipelineConfig):
        super().__init__(config)
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.post_init()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str | bytes | os.PathLike] = None,
        *model_args,
        config: Optional[GenatatorPipelineConfig] = None,
        **kwargs,
    ) -> "GenatatorPipelineModel":
        """Instantiate the lightweight wrapper without requiring checkpoint files.

        ``transformers.pipeline`` calls ``AutoModel.from_pretrained`` for the repo
        referenced by ``model=...``. For this scientific pipeline repo we only need
        configuration + remote code; the four task-specific models are loaded later
        by the pipeline class itself.
        """
        import os

        trust_remote_code = kwargs.pop("trust_remote_code", None)
        revision = kwargs.pop("revision", None)
        code_revision = kwargs.pop("code_revision", None)
        subfolder = kwargs.pop("subfolder", "")
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        local_files_only = kwargs.pop("local_files_only", False)
        token = kwargs.pop("token", None)
        kwargs.pop("use_safetensors", None)
        kwargs.pop("state_dict", None)
        kwargs.pop("from_tf", None)
        kwargs.pop("from_flax", None)
        kwargs.pop("weights_only", None)
        kwargs.pop("ignore_mismatched_sizes", None)
        kwargs.pop("low_cpu_mem_usage", None)
        kwargs.pop("device_map", None)
        requested_dtype = kwargs.pop("dtype", kwargs.pop("torch_dtype", None))

        if config is None:
            config = cls.config_class.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                code_revision=code_revision,
                subfolder=subfolder,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
            )

        model = cls(config, *model_args)
        if isinstance(requested_dtype, str):
            name = requested_dtype.lower().replace("torch.", "")
            if name == "auto":
                requested_dtype = torch.float32
            elif name in {"float16", "fp16", "half"}:
                requested_dtype = torch.float16
            elif name in {"bfloat16", "bf16"}:
                requested_dtype = torch.bfloat16
            elif name in {"float32", "fp32", "float"}:
                requested_dtype = torch.float32
            else:
                requested_dtype = None
        if isinstance(requested_dtype, torch.dtype):
            model = model.to(dtype=requested_dtype)
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutput:
        if input_ids is None:
            batch_size = 1
            seq_len = 1
            device = self.dummy_param.device
        else:
            batch_size = int(input_ids.shape[0])
            seq_len = int(input_ids.shape[1]) if input_ids.ndim > 1 else 1
            device = input_ids.device
        hidden = torch.zeros((batch_size, seq_len, 1), dtype=self.dummy_param.dtype, device=device)
        return BaseModelOutput(last_hidden_state=hidden)


try:
    AutoConfig.register("genatator_pipeline", GenatatorPipelineConfig)
except Exception:
    pass
