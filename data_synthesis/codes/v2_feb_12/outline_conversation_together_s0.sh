#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Step 1: generate outlines from cards
python -u "$SCRIPT_DIR/gen_outlines.py" --num-conversations 200 --seed 0

# Step 2: generate conversations from outlines
python "$SCRIPT_DIR/gen_conversations_shorter.py" \
    --outlines "$PROJECT_ROOT/outlines/outlines_n200_s0.jsonl"
