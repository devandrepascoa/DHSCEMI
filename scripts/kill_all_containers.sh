#!/usr/bin/env bash
# Kill and remove all running Docker containers
set -e

containers=$(docker ps -q)
if [ -z "$containers" ]; then
    echo "No running containers."
    exit 0
fi

echo "Killing $(echo "$containers" | wc -l) container(s)..."
docker kill $containers
echo "Done."
