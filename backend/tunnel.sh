#!/bin/sh
# Cortex Tunnels
# Ports:
# =======
# 2222 - ssh
# 6006 - Tensorboard
# 8080 - Frontend
# 8989 - Backend 

ssh -o ServerAliveInterval=60 \
-J zzy@snitch.ddns.net \
-L 2222:localhost:22 \
-L 6006:localhost:6006 \
-L 8989:localhost:8989 \
-N zzy@localhost -p 2222
