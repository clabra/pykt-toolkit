#!/bin/bash
# Auto-attach to pinn-dev container when opening a terminal in this project

# Check if we're already inside the container
if [ -f /.dockerenv ] || grep -q docker /proc/1/cgroup 2>/dev/null; then
    # Already inside container, activate virtual environment
    if [ -f /home/vscode/.pykt-env/bin/activate ]; then
        source /home/vscode/.pykt-env/bin/activate
    fi
    return 0 2>/dev/null || exit 0
fi

# Check if we're in the pykt-toolkit directory
if [[ "$PWD" == *"pykt-toolkit"* ]]; then
    # Check if container is running
    if docker ps --format '{{.Names}}' | grep -q '^pinn-dev$'; then
        echo "🐳 Attaching to pinn-dev container..."
        exec docker exec -it pinn-dev /bin/bash -c "cd /workspaces/pykt-toolkit && source /home/vscode/.pykt-env/bin/activate && exec bash"
    else
        echo "⚠️  Container 'pinn-dev' is not running."
        echo "Start it with: docker-compose up -d"
    fi
fi
