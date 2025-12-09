"""CUDA FP4 PerBlock quantization strategies"""

from typing import Any, Dict

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy


class CudaFp4EpLowLatencyStrategy(MoeStrategy):
    """CUDA FP4 EP low latency strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return DeepEpLowLatencyRouter(
            config,
            use_fp8_dispatch=False,  # FP4 doesn't use FP8 dispatch
            zero_copy=False,
            async_finish=False,
            return_recv_hook=False,
        )

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,  # FP4 is packed as uint8
            block_shape=[16, 16],  # Block shape for FP4 quantization (quant_blocksize=16)
        )

        return CutedslFp4Executor(
            config,
            weights,
            quant_config=quant_config,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutedslFp4Executor,
        )


class CudaFp4EpNormalStrategy(MoeStrategy):
    """CUDA FP4 EP normal mode strategy"""

    def create_router(self, config: GptInitModelParameters) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return DeepepNormalRouter(config)

    def create_executor(
        self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]
    ) -> Any:
        from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
            FusedMoEQuantConfig,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.uint8,  # FP4 is packed as uint8
            block_shape=[16, 16],  # Block shape for FP4 quantization (quant_blocksize=16)
        )

        return CutedslFp4Executor(
            config,
            weights,
            quant_config=quant_config,
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutedsl_fp4_executor import (
            CutedslFp4Executor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouter,
        )

        return StrategyAttributes(
            router_class=DeepepNormalRouter,
            executor_class=CutedslFp4Executor,
        )

