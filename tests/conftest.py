"""Конфигурация pytest для Meeting Recorder."""

import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
