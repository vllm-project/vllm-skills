"""
vLLM deployment utilities for local and Docker deployment.

Provides configuration and deployment classes for running vLLM
with OpenAI-compatible API.
"""

import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class VLLMConfig:
    """Configuration for vLLM server deployment.

    Attributes:
        model: Model name/path (HuggingFace model ID or local path)
        host: Host address to bind the server to
        port: Port number for the API server
        tensor_parallel_size: Number of GPUs for tensor parallelism
        gpu_memory_utilization: GPU memory utilization fraction (0.0-1.0)
        max_model_len: Maximum model length context window
        dtype: Data type for model weights (e.g., 'auto', 'half', 'bfloat16')
        quantization: Quantization method (e.g., 'awq', 'gptq', 'squeezellm')
        api_key: Optional API key for authentication
        enable_lora: Enable LoRA adapter support
        max_loras: Maximum number of LoRA adapters to load
        lora_extra_vocab_size: Maximum size of extra vocabulary per LoRA adapter
    """

    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    host: str = "0.0.0.0"
    port: int = 8000
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    quantization: Optional[str] = None
    api_key: Optional[str] = None
    enable_lora: bool = False
    max_loras: int = 8
    lora_extra_vocab_size: int = 256

    def to_command_args(self) -> List[str]:
        """Convert config to vLLM serve command line arguments.

        Returns:
            List of command-line arguments for vllm serve.
        """
        args = [
            "--model", self.model,
            "--host", self.host,
            "--port", str(self.port),
            "--tensor-parallel-size", str(self.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
        ]

        if self.max_model_len is not None:
            args.extend(["--max-model-len", str(self.max_model_len)])

        if self.dtype != "auto":
            args.extend(["--dtype", self.dtype])

        if self.quantization:
            args.extend(["--quantization", self.quantization])

        if self.api_key:
            args.extend(["--api-key", self.api_key])

        if self.enable_lora:
            args.extend([
                "--enable-lora",
                "--max-loras", str(self.max_loras),
                "--lora-extra-vocab-size", str(self.lora_extra_vocab_size),
            ])

        return args


@dataclass
class DeploymentResult:
    """Result of a deployment operation.

    Attributes:
        success: Whether the deployment was successful
        endpoint_url: URL of the deployed API endpoint
        instance_id: ID of the created instance (for cloud deployments)
        message: Human-readable status message
        metadata: Additional deployment metadata
    """

    success: bool
    endpoint_url: Optional[str] = None
    instance_id: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VLLMDeployer:
    """Main class for deploying vLLM online services.

    This class provides a unified interface for deploying vLLM with
    OpenAI-compatible API to local or Docker environments.

    Example:
        >>> from vllm_deploy import VLLMConfig, VLLMDeployer
        >>> config = VLLMConfig(model="Qwen/Qwen2.5-1.5B-Instruct")
        >>> deployer = VLLMDeployer(vllm_config=config)
        >>> result = deployer.deploy_local()
        >>> print(f"API running at: {result.endpoint_url}")
    """

    def __init__(
        self,
        vllm_config: Optional[VLLMConfig] = None,
    ):
        """Initialize the VLLMDeployer.

        Args:
            vllm_config: Configuration for the vLLM server
        """
        self.vllm_config = vllm_config or VLLMConfig()
        self._deployment = None

    def deploy_local(self, background: bool = True) -> DeploymentResult:
        """Deploy vLLM as a local server.

        Args:
            background: Run the server in the background

        Returns:
            DeploymentResult with endpoint URL and status
        """
        import subprocess

        cmd = ["vllm", "serve"] + self.vllm_config.to_command_args()

        try:
            if background:
                # Run in background using subprocess
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True
                )
                self._deployment = process
                time.sleep(2)  # Give server time to start

                # Check if process is still running
                if process.poll() is None:
                    endpoint_url = f"http://{self.vllm_config.host}:{self.vllm_config.port}/v1"
                    return DeploymentResult(
                        success=True,
                        endpoint_url=endpoint_url,
                        message=f"vLLM server started locally on port {self.vllm_config.port}",
                        metadata={"pid": process.pid, "cmd": " ".join(cmd)}
                    )
                else:
                    # Process terminated, get error
                    _, stderr = process.communicate()
                    return DeploymentResult(
                        success=False,
                        message=f"Server failed to start: {stderr.decode()}"
                    )
            else:
                # Run in foreground (blocking)
                subprocess.run(cmd, check=True)
                return DeploymentResult(
                    success=True,
                    message="Server ran successfully",
                )

        except FileNotFoundError:
            return DeploymentResult(
                success=False,
                message="vLLM not installed. Run: pip install vllm"
            )
        except Exception as e:
            return DeploymentResult(
                success=False,
                message=f"Deployment failed: {str(e)}"
            )

    def deploy_docker(
        self,
        image: str = "vllm/vllm-openai:latest",
        port_mapping: Optional[int] = None,
        background: bool = True,
    ) -> DeploymentResult:
        """Deploy vLLM using Docker.

        Args:
            image: Docker image to use
            port_mapping: Host port for the container. If None, uses vllm_config.port.
            background: Run container in background (detached mode)

        Returns:
            DeploymentResult with container info and endpoint URL
        """
        import subprocess

        if port_mapping is None:
            port_mapping = self.vllm_config.port

        # Build docker run command
        docker_cmd = [
            "docker", "run",
            "--gpus", "all",
            "-p", f"{port_mapping}:{self.vllm_config.port}",
            "--shm-size", "10g",  # Shared memory size for model loading
        ]

        if background:
            docker_cmd.append("-d")

        # Add environment variables for vLLM config
        env_args = self._build_docker_env_args()
        docker_cmd.extend(env_args)

        # Image and model
        docker_cmd.append(image)
        docker_cmd.append("--model")
        docker_cmd.append(self.vllm_config.model)

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            endpoint_url = f"http://localhost:{port_mapping}/v1"

            return DeploymentResult(
                success=True,
                endpoint_url=endpoint_url,
                message="vLLM deployed in Docker container",
                metadata={
                    "container_id": result.stdout.strip() if background else None,
                    "image": image,
                }
            )

        except FileNotFoundError:
            return DeploymentResult(
                success=False,
                message="Docker not found. Please install Docker."
            )
        except subprocess.CalledProcessError as e:
            return DeploymentResult(
                success=False,
                message=f"Docker deployment failed: {e.stderr}"
            )

    def _build_docker_env_args(self) -> list:
        """Build environment variable arguments for Docker.

        Returns:
            List of -e flags for docker run.
        """
        env_args = []

        if self.vllm_config.api_key:
            env_args.extend(["-e", f"VLLM_API_KEY={self.vllm_config.api_key}"])

        if self.vllm_config.tensor_parallel_size > 1:
            env_args.extend(["-e", f"TENSOR_PARALLEL_SIZE={self.vllm_config.tensor_parallel_size}"])

        env_args.extend([
            "-e", f"GPU_MEMORY_UTILIZATION={self.vllm_config.gpu_memory_utilization}",
        ])

        return env_args

    def stop(self) -> bool:
        """Stop the currently running deployment.

        Returns:
            True if stopped successfully, False otherwise
        """
        if self._deployment is None:
            return False

        try:
            self._deployment.terminate()
            self._deployment.wait(timeout=30)
            return True
        except Exception:
            return False

    def health_check(self, endpoint: Optional[str] = None) -> bool:
        """Check if the vLLM server is healthy.

        Args:
            endpoint: URL to check (uses deployed endpoint if None)

        Returns:
            True if server is healthy, False otherwise
        """
        import urllib.request

        if endpoint is None:
            endpoint = f"http://{self.vllm_config.host}:{self.vllm_config.port}/v1/models"

        try:
            with urllib.request.urlopen(endpoint, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
