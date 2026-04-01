---
name: vllm-deploy-k8s
description: Deploy vLLM to Kubernetes (K8s) with GPU support, health probes, and OpenAI-compatible API endpoint. Use this skill whenever the user wants to deploy, run, or serve vLLM on a Kubernetes cluster, including creating deployments, services, checking existing deployments, or managing vLLM on K8s.
---

# vLLM Kubernetes Deployment

A Claude skill for deploying vLLM to Kubernetes using YAML templates. Deploys a vLLM OpenAI-compatible server as a Kubernetes Deployment with a ClusterIP Service, GPU resources, and health probes.

## What this skill does

- Deploy vLLM as a Kubernetes Deployment + Service with NVIDIA GPU support
- Check if a vLLM deployment already exists before deploying
- Check if the Hugging Face token secret exists, and ask the user for their token if not
- Use the `vllm/vllm-openai:latest` image by default (user can specify a different version)
- Provide sensible default configuration that users can customize (model, replicas, GPU count, extra vLLM flags, etc.)

## Prerequisites

- `kubectl` configured with access to a Kubernetes cluster
- NVIDIA GPU Operator or device plugin installed on cluster nodes
- Hugging Face token (required for gated models like Llama, optional for public models)

## Deployment Steps

### Step 1: Check HF token secret

Before deploying, check if the `hf-token` Kubernetes secret exists in the target namespace:

```bash
kubectl get secret hf-token -n <namespace>
```

- If the secret **exists**: proceed to Step 2.
- If the secret **does not exist**: ask the user to provide their Hugging Face token, then create the secret:

```bash
kubectl create secret generic hf-token --from-literal=HF_TOKEN="<user-provided-token>" -n <namespace>
```

This is required for gated models (e.g., `meta-llama/Meta-Llama-3.1-8B`). For public models, the secret is optional but recommended to avoid rate limits.

### Step 2: Check if deployment already exists

Before applying, check if a vLLM deployment already exists:

```bash
kubectl get deployment vllm -n <namespace>
```

- If it **exists**: inform the user that the deployment already exists. Show the current image and status. Ask the user if they want to update it or skip.
- If it **does not exist**: proceed to deploy.

### Step 3: Deploy

Apply the template YAML files to deploy vLLM:

```bash
kubectl apply -f templates/vllm-service.yaml -n <namespace>
kubectl apply -f templates/vllm-deployment.yaml -n <namespace>
```

### Step 4: Wait and verify

Wait for the deployment to roll out:

```bash
kubectl rollout status deployment/vllm -n <namespace> --timeout=600s
```

Verify the pod is running and ready:

```bash
kubectl get pods -n <namespace> -l app=vllm
```

Confirm the pod shows `READY 1/1` and `STATUS Running`. If the pod is not ready yet, wait and check again. If it's in `CrashLoopBackOff` or `Error`, check the logs with `kubectl logs -n <namespace> -l app=vllm`.

### Step 5: Print deployment summary

Once the pod is ready, print a summary message to the user in this format (replace placeholders with actual values):

```
🎉 **vLLM Deployment Successful!**

| Resource | Name | Status |
|----------|------|--------|
| Deployment | <deployment-name> | <ready>/<total> Ready |
| Service | <service-name> | ClusterIP:<port> |
| Pod | <pod-name> | Running |
| Image | <image> | |
| Model | <model> | |

&nbsp;

**To test the API, run these two commands in your terminal:**

**1. Open a port-forward** (this connects your local port <port> to the vLLM service inside the cluster):

kubectl port-forward svc/vllm-svc <port>:<port> -n <namespace>

**2. In a separate terminal**, send a test request to the OpenAI-compatible API:

curl -s http://localhost:<port>/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model>","messages":[{"role":"user","content":"Hello!"}],"max_tokens":50}' | python3 -m json.tool

If everything is working, you'll get a JSON response with the model's reply.
```

## Default Configuration

The templates use the following defaults:

| Parameter | Default Value |
|-----------|---------------|
| Image | `vllm/vllm-openai:latest` |
| Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Port | `8000` |
| Replicas | `1` |
| GPU count | `1` |
| GPU memory utilization | `0.85` |
| Tensor parallel size | `1` |
| CPU request / limit | `12` / `128` |
| Memory request / limit | `100Gi` / `400Gi` |
| Shared memory (dshm) | `80Gi` |

## Customization

When the user requests changes, modify the template YAML files before applying. The following can be customized:

- **Image version**: Change `image: vllm/vllm-openai:<version>` in `templates/vllm-deployment.yaml` (default: `latest`). Use a specific version tag like `v0.17.1` if the user requests it.
- **Model**: Change the model name in the `vllm serve` command inside the Deployment `args`.
- **Extra vLLM flags**: Append additional flags to the `vllm serve` command in the Deployment `args` (e.g., `--max-model-len 4096`, `--kv-cache-dtype fp8`, `--enforce-eager`, `--generation-config vllm`).
- **Replicas**: Change `replicas:` in the Deployment spec.
- **GPU count**: Change `nvidia.com/gpu` in both `requests` and `limits` under resources.
- **Tensor parallel size**: Change `--tensor-parallel-size` flag to match the GPU count.
- **CPU/Memory resources**: Change `cpu` and `memory` values under `requests` and `limits`.
- **Port**: Change `containerPort` in the Deployment, `port`/`targetPort` in the Service, the `port` in all health probes (liveness, readiness, startup), AND add `--port <port>` to the `vllm serve` command in args. All four must match.
- **Namespace**: Apply to a specific namespace using `-n <namespace>`.
- **Shared memory size**: Change the `sizeLimit` of the `dshm` emptyDir volume.

Edit the template files using the Edit tool, then apply the modified templates.

## Status Check

```bash
kubectl get deployment,svc,pods -n <namespace> -l app=vllm
```

## Cleanup

When the user asks to clean up or delete the vLLM deployment, run the following steps:

1. Delete the Deployment and Service:

```bash
kubectl delete -f templates/vllm-deployment.yaml -n <namespace>
kubectl delete -f templates/vllm-service.yaml -n <namespace>
```

2. Ask the user if they also want to delete the HF token secret. If yes:

```bash
kubectl delete secret hf-token -n <namespace>
```

3. Verify everything is cleaned up:

```bash
kubectl get deployment,svc,pods -n <namespace> -l app=vllm
```

4. Print a summary message to the user:

```
vLLM deployment has been cleaned up from namespace <namespace>.
Deleted: Deployment/vllm, Service/vllm-svc
HF token secret: <kept/deleted>
```

## Troubleshooting

- **Pod stuck in Pending**: No GPU nodes available. Check `kubectl describe pod <pod-name>` for scheduling errors. Ensure NVIDIA GPU Operator or device plugin is installed.
- **Pod OOMKilled**: Increase `memory` limits in the Deployment, or use a smaller model.
- **ImagePullBackOff**: Check the image name and tag. Verify the node has access to Docker Hub / the container registry.
- **Startup probe failures (CrashLoopBackOff)**: Model download may be slow. Check logs with `kubectl logs <pod-name>`. Ensure `hf-token` secret exists for gated models. Increase `failureThreshold` on the startup probe if needed.
- **HF_TOKEN not working**: Verify the secret exists: `kubectl get secret hf-token -n <namespace>`. Check the token is valid.
- **GPU not detected in container**: Ensure `nvidia.com/gpu` resource is requested and the NVIDIA device plugin is running on the node.

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Docker Images](https://hub.docker.com/r/vllm/vllm-openai/tags)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
- [Kubernetes GPU Scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/)
