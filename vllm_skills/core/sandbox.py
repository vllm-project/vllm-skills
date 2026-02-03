"""Sandbox interfaces for safe code execution"""

import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import Any


class SandboxInterface(ABC):
    """Abstract interface for execution sandboxes"""

    @abstractmethod
    async def execute(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """
        Execute code in a sandbox

        Args:
            code: Code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Dictionary with execution results:
                - stdout: Standard output
                - stderr: Standard error
                - return_code: Exit code
                - error: Error message if any
        """
        pass


class LocalSandbox(SandboxInterface):
    """Execute code in local process (less secure)"""

    async def execute(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """
        Execute code in local subprocess

        Args:
            code: Code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Execution results dictionary
        """
        try:
            # Execute in subprocess with timeout
            process = await asyncio.create_subprocess_shell(
                code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return {
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "return_code": process.returncode,
                    "error": None,
                }
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1,
                    "error": f"Execution timeout after {timeout} seconds",
                }

        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "error": f"Execution error: {str(e)}",
            }


class DockerSandbox(SandboxInterface):
    """Execute code in Docker container (more secure)"""

    def __init__(self, image: str = "python:3.11-slim"):
        """
        Initialize Docker sandbox

        Args:
            image: Docker image to use
        """
        self.image = image

    async def execute(self, code: str, timeout: int = 30) -> dict[str, Any]:
        """
        Execute code in Docker container

        Args:
            code: Code to execute
            timeout: Maximum execution time in seconds

        Returns:
            Execution results dictionary
        """
        try:
            # Build docker run command
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "--network=none",  # No network access
                "--memory=512m",  # Memory limit
                "--cpus=1.0",  # CPU limit
                self.image,
                "sh",
                "-c",
                code,
            ]

            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return {
                    "stdout": stdout.decode() if stdout else "",
                    "stderr": stderr.decode() if stderr else "",
                    "return_code": process.returncode,
                    "error": None,
                }
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "stdout": "",
                    "stderr": "",
                    "return_code": -1,
                    "error": f"Execution timeout after {timeout} seconds",
                }

        except FileNotFoundError:
            return {
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "error": "Docker not found. Please install Docker.",
            }
        except Exception as e:
            return {
                "stdout": "",
                "stderr": "",
                "return_code": -1,
                "error": f"Docker execution error: {str(e)}",
            }
