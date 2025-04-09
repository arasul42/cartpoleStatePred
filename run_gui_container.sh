#!/bin/bash

# Allow local Docker access to X11
xhost +local:docker

# Option 1: Use docker-compose (optional, uncomment if using docker-compose)
# docker compose up -d

# Option 2: Run container directly
docker exec -it ubuntu_gym /bin/bash

# (Optional) Revoke access after container exits
xhost -local:docker
