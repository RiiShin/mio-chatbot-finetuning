# Mio Chatbot Finetuning

Code for synthesizing SFT training data for a locally deployed character chatbot called **Mio** (星川 澪).

## Directory Structure

```
data_synthesis/
├── codes/
│   ├── v2_feb_12/                        # Data generation pipeline
│   │   ├── gen_conversations_shorter.py  # Generate conversations from outlines
│   │   └── outline_conversation_together_s0.sh  # Example run script
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

### Generate Conversations

```bash
pip install openai tqdm

export OPENAI_API_KEY="sk-..."

cd data_synthesis/codes/v2_feb_12
python gen_conversations_shorter.py \
    --outlines ../../outlines/outlines_n200_s0.jsonl
```

### Expand Cards (Optional)

```bash
export OPENAI_API_KEY="sk-..."

python codes/card_prepare/extend_user.py
python codes/card_prepare/extend_mio.py
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
