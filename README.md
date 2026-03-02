# Mio Chatbot Finetuning

Code for synthesizing SFT training data for a locally deployed character chatbot called **Mio** (星川 澪).

## Directory Structure

```
data_synthesis/
├── codes/
│   ├── v2_feb_12/                        # Data generation pipeline
│   │   ├── gen_outlines.py               # Step 1: generate conversation outlines from cards
│   │   ├── gen_conversations_shorter.py  # Step 2: generate full conversations from outlines
│   │   └── outline_conversation_together_s0.sh  # Example run script (both steps)
│   └── card_prepare/                     # Card expansion utilities (L1 → L2)
│       ├── extend_user.py                # Expand user-side cards
│       ├── extend_mio.py                 # Expand Mio-side cards
│       └── convert_tsv.py                # Convert txt cards to csv
├── card_sets/                            # Character card materials
│   ├── l1/                               # Base cards (csv + original txt)
│   ├── l2/                               # Expanded cards (generated)
│   ├── 基础setting.txt                    # Base character setting
│   └── 腹黑setting.txt                    # Dark character setting
├── sft_system_prompt.txt                 # System prompt for SFT data
└── v2_convs/                             # Generated conversation data
    ├── conversations_n200_s0_shorter.jsonl
    └── conversations_n200_s1_shorter.jsonl
```

## Quick Start

```bash
pip install openai tqdm scikit-learn numpy

export OPENAI_API_KEY="sk-..."
```

### Full Pipeline (Step 1 + Step 2)

```bash
# Run both steps together
cd data_synthesis/codes/v2_feb_12
bash outline_conversation_together_s0.sh
```

Or run each step separately:

### Step 1 — Generate Outlines

Samples from card_sets (user_profiles, scene, refusal) and asks the LLM to write conversation outlines.

```bash
python data_synthesis/codes/v2_feb_12/gen_outlines.py \
    --num-conversations 200 --seed 0
```

Output: `data_synthesis/outlines/outlines_n200_s0.jsonl`

### Step 2 — Generate Conversations

Reads outlines and generates full multi-turn conversations.

```bash
python data_synthesis/codes/v2_feb_12/gen_conversations_shorter.py \
    --outlines data_synthesis/outlines/outlines_n200_s0.jsonl
```

Output: `data_synthesis/v2_convs/conversations_n200_s0_shorter.jsonl`

### Expand Cards (Optional)

Expand L1 base cards to L2 detail cards using GPT:

```bash
python data_synthesis/codes/card_prepare/extend_user.py
python data_synthesis/codes/card_prepare/extend_mio.py
```

## Pre-generated Data

`data_synthesis/v2_convs/` contains two ready-to-use conversation datasets (seed 0 and seed 1, 200 conversations each) in JSONL format. Each line contains:

```json
{
  "conv_id": 0,
  "system": "...",
  "conversation": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```
