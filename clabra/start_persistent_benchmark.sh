#!/bin/bash
# Persistent Benchmark Runner Script

SCRIPT_DIR="/workspaces/pykt-toolkit/assistant"
VENV_PATH="/home/vscode/.pykt-env/bin/activate"
BENCHMARK_SCRIPT="run_transformer_attention_benchmark.py"
LOG_FILE="benchmark_persistent_$(date +%Y%m%d_%H%M%S).log"

cd "$SCRIPT_DIR"
source "$VENV_PATH"

echo "🚀 Starting persistent transformer benchmark..."
echo "📝 Log file: $LOG_FILE"
echo "⏰ Started at: $(date)"

# Run with nohup for persistence
nohup python "$BENCHMARK_SCRIPT" > "$LOG_FILE" 2>&1 &

BENCHMARK_PID=$!
echo "🔢 Process ID: $BENCHMARK_PID"
echo "$BENCHMARK_PID" > benchmark.pid

echo "✅ Benchmark started successfully!"
echo "📊 Monitor with: tail -f $SCRIPT_DIR/$LOG_FILE"
echo "🛑 Stop with: kill $BENCHMARK_PID"