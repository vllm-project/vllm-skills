#!/bin/bash

# vLLM Quickstart Script
# Installs vLLM, starts server with Qwen2.5-1.5B-Instruct, and tests the API

set -e

# Default configuration
MODEL="Qwen/Qwen2.5-1.5B-Instruct"
PORT=8000
HOST="0.0.0.0"
VENV_PATH="."
MAX_WAIT=120    # Maximum seconds to wait for server startup
VRAM=0.8        # A conservative default for GPU memory utilization to avoid potential OOM issues. vLLM uses 0.9 by default.

# Parse arguments
COMMAND=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        --gpu_memory_utilization)
            VRAM="$2"
            shift 2
            ;;
        install|start|stop|test|status|restart|all)
            COMMAND="$1"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [command] [--model MODEL] [--port PORT] [--venv VENV_PATH] [--gpu_memory_utilization VRAM]"
            exit 1
            ;;
    esac
done

# Set default command if none provided
COMMAND="${COMMAND:-all}"

# Set log and pid file paths (handle trailing slash)
VENV_PATH="${VENV_PATH%/}"  # Remove trailing slash if present
LOG_FILE="$VENV_PATH/tmp/vllm-server.log"
PID_FILE="$VENV_PATH/tmp/vllm-server.pid"

# Create log directory if it doesn't exist
LOG_DIR="$(dirname "$LOG_FILE")"
if [[ ! -d "$LOG_DIR" ]]; then
    mkdir -p "$LOG_DIR"
fi

# Validate and activate virtual environment
if [[ -n "$VENV_PATH" ]] && [[ "$VENV_PATH" != "." ]]; then
    # Check if the virtual environment directory exists
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "Error: Virtual environment path '$VENV_PATH' does not exist."
        exit 1
    fi

    # Determine the activation script based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Linux or macOS
        ACTIVATE_SCRIPT="$VENV_PATH/bin/activate"
    else
        # Windows (Git Bash or WSL)
        ACTIVATE_SCRIPT="$VENV_PATH/Scripts/activate"
    fi

    # Check if the activation script exists
    if [[ ! -f "$ACTIVATE_SCRIPT" ]]; then
        echo "Error: Activation script not found at '$ACTIVATE_SCRIPT'."
        echo "Please ensure this is a valid virtual environment."
        exit 1
    fi

    # Activate the virtual environment
    source "$ACTIVATE_SCRIPT"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    printf "${BLUE}[INFO]${NC} %s\n" "$1"
}

log_success() {
    printf "${GREEN}[SUCCESS]${NC} %s\n" "$1"
}

log_warning() {
    printf "${YELLOW}[WARNING]${NC} %s\n" "$1"
}

log_error() {
    printf "${RED}[ERROR]${NC} %s\n" "$1"
}

# Check if server is running
is_server_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Detect hardware backend
detect_backend() {
    if command -v nvidia-smi &> /dev/null; then
        echo "NVIDIA CUDA"
    elif [[ -f "/dev/kfd" ]] && [[ -d "/dev/dri" ]]; then
        echo "AMD ROCm"
    elif [[ -n "$TPU_NAME" ]] || command -v gcloud &> /dev/null; then
        echo "Google TPU"
    else
        echo "CPU"
    fi
}

# Install vLLM and dependencies
install_vllm() {
    log_info "Installing vLLM and dependencies..."

    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    log_info "Python version: $PYTHON_VERSION"

    # Detect hardware backend
    BACKEND=$(detect_backend)
    log_info "Detected backend: $BACKEND"

    # Check if uv is available, otherwise use pip
    if command -v uv &> /dev/null; then
        log_info "Using uv package manager..."
        PACKAGE_MANAGER="uv pip"
    else
        log_info "Using pip package manager..."
        PACKAGE_MANAGER="pip"
    fi

    # Install vLLM based on detected backend
    case $BACKEND in
        "NVIDIA CUDA")
            log_info "Installing vLLM for NVIDIA CUDA..."
            $PACKAGE_MANAGER install vllm openai
            ;;
        "AMD ROCm")
            log_info "Installing vLLM for AMD ROCm..."
            $PACKAGE_MANAGER install vllm openai
            ;;
        "Google TPU")
            log_info "Installing vLLM for Google TPU..."
            $PACKAGE_MANAGER install vllm-tpu openai
            ;;
        *)
            log_info "Installing vLLM for CPU..."
            $PACKAGE_MANAGER install vllm openai
            ;;
    esac

    log_success "vLLM installation complete!"

    # Verify installation
    python3 -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
        log_error "vLLM installation verification failed"
        exit 1
    }
}

# Start vLLM server
start_server() {
    if is_server_running; then
        log_warning "Server is already running (PID: $(cat $PID_FILE))"
        return 0
    fi

    log_info "Starting vLLM server with model: $MODEL"
    log_info "Server will be available at: http://localhost:$PORT"
    log_info "Logs: $LOG_FILE"

    # Start server in background
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --gpu_memory_utilization "$VRAM" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    log_success "Server started with PID: $(cat $PID_FILE)"

    # Wait for server to be ready
    log_info "Waiting for server to be ready..."
    ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            log_success "Server is ready!"
            return 0
        fi
        sleep 2
        ELAPSED=$((ELAPSED + 2))
        echo -n "."
    done

    echo ""
    log_error "Server failed to start within ${MAX_WAIT}s. Check logs: $LOG_FILE"
    exit 1
}

# Stop vLLM server
stop_server() {
    if ! is_server_running; then
        log_warning "Server is not running"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    log_info "Stopping server (PID: $PID)..."
    kill "$PID" 2>/dev/null || true

    # Wait for process to stop
    for i in {1..10}; do
        if ! ps -p "$PID" > /dev/null 2>&1; then
            rm -f "$PID_FILE"
            log_success "Server stopped"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    log_warning "Force killing server..."
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    log_success "Server stopped"
}

# Test the API
test_api() {
    log_info "Testing OpenAI-compatible API..."

    # Check if server is running
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        log_error "Server is not responding. Make sure it's running with: $0 start"
        exit 1
    fi

    # Test /v1/models endpoint
    log_info "Fetching available models..."
    MODELS_RESPONSE=$(curl -s "http://localhost:$PORT/v1/models")
    echo "$MODELS_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$MODELS_RESPONSE"

    echo ""
    log_info "Sending chat completion request..."

    # Test chat completion
    RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello and introduce yourself in one sentence.\"}],
            \"max_tokens\": 50,
            \"temperature\": 0.7
        }")

    # Pretty print response
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"

    # Extract and display just the message content
    CONTENT=$(echo "$RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null)

    if [ -n "$CONTENT" ]; then
        echo ""
        log_success "API Test Successful!"
        printf "${GREEN}Response:${NC} %s\n" "$CONTENT"
    else
        log_error "Failed to get valid response from API"
        exit 1
    fi
}

# Show server status
show_status() {
    echo "=== vLLM Server Status ==="
    echo ""

    if is_server_running; then
        PID=$(cat "$PID_FILE")
        log_success "Server is running (PID: $PID)"
        echo "  Model: $MODEL"
        echo "  Endpoint: http://localhost:$PORT"
        echo "  Health: http://localhost:$PORT/health"
        echo "  Logs: $LOG_FILE"
        echo ""

        # Show last few log lines
        if [ -f "$LOG_FILE" ]; then
            echo "Recent logs:"
            tail -n 5 "$LOG_FILE"
        fi
    else
        log_warning "Server is not running"
        echo ""
        echo "Start the server with: $0 start"
    fi
}

# Main workflow
run_all() {
    log_info "Running complete vLLM quickstart workflow..."
    echo ""

    # Step 1: Install
    install_vllm
    echo ""

    # Step 2: Start server
    start_server
    echo ""

    # Step 3: Test API
    test_api
    echo ""

    # Step 4: Show status
    show_status
    echo ""

    log_success "Quickstart complete! Server is running at http://localhost:$PORT"
    log_info "To stop the server, run: $0 stop"
}

# Execute command
case "$COMMAND" in
    install)
        install_vllm
        ;;
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    test)
        test_api
        ;;
    status)
        show_status
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: $0 [command] [--model MODEL] [--port PORT] [--venv VENV_PATH] [--gpu_memory_utilization VRAM]"
        echo ""
        echo "Commands:"
        echo "  install  - Install vLLM and dependencies"
        echo "  start    - Start the vLLM server"
        echo "  stop     - Stop the vLLM server"
        echo "  test     - Test the OpenAI-compatible API"
        echo "  status   - Show server status"
        echo "  restart  - Restart the server"
        echo "  all      - Run complete workflow (default)"
        echo ""
        echo "Options:"
        echo "  --model MODEL                   Model to use (default: Qwen/Qwen2.5-1.5B-Instruct)"
        echo "  --port PORT                     Port to run server on (default: 8000)"
        echo "  --venv VENV_PATH                Virtual environment path (default: .)"
        echo "  --gpu_memory_utilization VRAM   GPU memory utilization (default: 0.8)"
        exit 1
        ;;
esac
