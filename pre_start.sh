#!/usr/bin/env bash
export PYTHONUNBUFFERED=1

echo "Container is running"

# Sync venv to workspace to support Network volumes
echo "Syncing venv to workspace, please wait..."
rsync -au /venv/ /workspace/venv/
rm -rf /venv

# Sync Web UI to workspace to support Network volumes
echo "Syncing Stable Diffusion to workspace, please wait..."
rsync -au /app/ /workspace/app/
rm -rf /app

# Configure accelerate
echo "Configuring accelerate..."
mkdir -p /root/.cache/huggingface/accelerate
mv /accelerate.yaml /root/.cache/huggingface/accelerate/default_config.yaml

# Create logs directory
mkdir -p /workspace/logs

if [[ ${DISABLE_AUTOLAUNCH} ]]
then
    echo "Auto launching is disabled so the applications will not be started automatically"
    echo "You can launch them manually using the launcher scripts:"
    echo ""
    echo "   Stable Diffusion Web UI:"
    echo "   ---------------------------------------------"
    echo "   cd /workspace/stable-diffusion-webui"
    echo "   deactivate && source /workspace/venv/bin/activate"
    echo "   ./sd.sh -f"
    echo ""
else
    echo "Starting Stable Diffusion"
    cd /workspace/app
    nohup ./sd.sh -f > /workspace/logs/sd.log 2>&1 &
    echo "Stable Diffusion started"
    echo "Log file: /workspace/logs/sd.log"
fi

echo "All services have been started"