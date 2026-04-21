#!/bin/bash
exec gunicorn app:app --workers=1 --timeout=300 --bind=0.0.0.0:$PORT --chdir /opt/render/project/src
