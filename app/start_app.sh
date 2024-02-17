#!/bin/bash
cd /app
# Запустите сервер FastAPI
uvicorn main:app --host 0.0.0.0 --port 3001 &
# Запустите nginx
nginx -g 'daemon off;'