from typing import List, Optional

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.logger import init_logger
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import ExecuteModelRequest
from vllm.utils import make_async
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class NPUExecutor(GPUExecutor):

    def _init_executor(self) -> None:
        # TODO: 这两个参数作用
        assert not self.scheduler_config.chunked_prefill_enabled, (
            "Chunked prefill is not yet supported for NPU backend")
        assert not self.speculative_config, (
            "Speculative decoding is not yet supported for NPU backend")

        # Instantiate the worker and load the model to the device.
        self.driver_worker = self._create_worker()
        self.driver_worker.init_device()
        self.driver_worker.load_model()

    def _create_worker(self,
                       local_rank: int = 0,
                       rank: int = 0,
                       distributed_init_method: Optional[str] = None):
        worker_module_name = "vllm.worker.npu_worker"
        worker_class_name = "NPUWorker"

        wrapper = WorkerWrapperBase(
            worker_module_name=worker_module_name,
            worker_class_name=worker_class_name,
        )
        wrapper.init_worker(**self._get_worker_kwargs(local_rank, rank,
                                                      distributed_init_method))
        return wrapper.worker

    # 输出格式是否要改变？？
    def execute_model(
            self,
            execute_model_req: ExecuteModelRequest) -> List[SamplerOutput]:
        output = self.driver_worker.execute_model(execute_model_req)
        return output


class NPUExecutorAsync(NPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[SamplerOutput]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output
