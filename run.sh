#!/bin/bash
USE_LOCAL_RENDERER=true uvicorn main:app --host 0.0.0.0 --port 8000 --reload
