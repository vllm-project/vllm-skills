# Common vLLM Deployment Issues

## Installation Issues

### Issue: vLLM installation fails
**Symptoms**: `pip install vllm` errors out

**Solutions**:
1. Check Python version (requires Python 3.9+)
   ```bash
   python --version
   ```

2. Check CUDA version compatibility
   ```bash
   nvidia-smi
   ```

3. Install from source if binary fails
   ```bash
   git clone https://github.com/vllm-project/vllm.git
   cd vllm
   pip install -e .
   ```

## Model Loading Issues

### Issue: "Model not found" error
**Symptoms**: vLLM cannot find or load the model

**Solutions**:
1. Verify model path or HuggingFace ID
2. Check HuggingFace authentication (for gated models)
   ```bash
   huggingface-cli login
   ```
3. Pre-download model
   ```bash
   huggingface-cli download model-name
   ```

### Issue: Slow model loading
**Symptoms**: Model takes very long to load

**Solutions**:
1. Download model locally first
2. Use faster storage (NVMe SSD)
3. Enable model caching in vLLM

## Performance Issues

### Issue: Low throughput
**Symptoms**: Requests take too long to complete

**Solutions**:
1. Increase `max_num_seqs` for batch processing
2. Enable prefix caching: `--enable-prefix-caching`
3. Use tensor parallelism on multi-GPU systems
4. Adjust `gpu_memory_utilization` (try 0.95)

### Issue: High latency
**Symptoms**: First token latency is too high

**Solutions**:
1. Reduce `max_num_seqs` for smaller batches
2. Disable chunked prefill
3. Use smaller models or quantization
4. Optimize prompt length

## Server Issues

### Issue: Server crashes unexpectedly
**Symptoms**: vLLM server terminates without clear error

**Solutions**:
1. Check logs for OOM errors (see memory_issues.md)
2. Verify CUDA compatibility (see cuda_errors.md)
3. Update vLLM to latest version
4. Check system resources (RAM, disk space)

### Issue: "Port already in use"
**Symptoms**: Cannot start server on default port

**Solutions**:
1. Use different port: `--port 8001`
2. Kill process using port:
   ```bash
   lsof -ti:8000 | xargs kill -9
   ```

## API Issues

### Issue: Requests timeout
**Symptoms**: API requests hang or timeout

**Solutions**:
1. Increase request timeout
2. Check if model is still loading
3. Verify server is running: `curl http://localhost:8000/health`
4. Check server logs for errors

### Issue: Invalid responses
**Symptoms**: API returns malformed or unexpected responses

**Solutions**:
1. Verify request format matches OpenAI API
2. Check model's chat template compatibility
3. Validate JSON in request payload
4. Review vLLM server logs

## Multi-GPU Issues

### Issue: Tensor parallelism not working
**Symptoms**: vLLM uses only one GPU

**Solutions**:
1. Verify `--tensor-parallel-size` matches GPU count
2. Check NCCL installation
3. Ensure GPUs are on same node
4. Set environment variable: `export NCCL_DEBUG=INFO`

### Issue: GPU communication errors
**Symptoms**: Errors during multi-GPU inference

**Solutions**:
1. Update NCCL: `pip install --upgrade nccl-cu12`
2. Use `--disable-custom-all-reduce` flag
3. Check GPU interconnect (NVLink preferred)
4. Verify all GPUs are visible: `nvidia-smi`

## See Also
- [CUDA Errors](cuda_errors.md)
- [Memory Issues](memory_issues.md)
- [Kernel Issues](kernel_issues.md)
