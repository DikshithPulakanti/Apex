# training/generate_dataset.py
# Generates synthetic hypothesis dataset for fine-tuning

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import time
import random
import anthropic
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# ── Research domains APEX covers ──────────────────────────────────────────

DOMAINS = [
    "machine learning", "natural language processing",
    "computer vision", "reinforcement learning",
    "graph neural networks", "protein folding",
    "drug discovery", "materials science",
    "robotics", "quantum computing",
    "federated learning", "neural architecture search",
    "explainable AI", "multimodal learning",
    "knowledge graphs", "autonomous systems",
    "generative models", "causal inference",
    "computational biology", "speech recognition"
]

FLAW_TYPES = [
    "unfalsifiable",      # can't be tested or disproven
    "too_vague",          # no specific measurable claim
    "circular_reasoning", # conclusion hidden in premise
    "no_mechanism",       # states outcome without explaining how
    "already_proven",     # restates known facts as novel
    "untestable_scope",   # requires impossible experiments
]


# ── Generate valid hypotheses ─────────────────────────────────────────────

def generate_valid_batch(domain: str, batch_size: int = 10) -> list:
    """Ask Claude to generate scientifically valid hypotheses."""

    prompt = f"""Generate {batch_size} scientifically valid, novel research hypotheses 
in the domain of {domain}.

Each hypothesis must be:
- Specific and falsifiable
- Novel (not restating known facts)
- Testable with current technology
- Between 1-3 sentences

Return ONLY a JSON array of objects, no markdown, no explanation:
[
  {{"hypothesis": "...", "testability": 0.0-1.0}},
  ...
]

testability = how easily this could be tested (0 = impossible, 1 = straightforward experiment)"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()
        # Clean potential markdown fences
        text = text.replace("```json", "").replace("```", "").strip()
        hypotheses = json.loads(text)

        for h in hypotheses:
            h['label'] = 1
            h['flaw_type'] = None
            h['domain'] = domain

        return hypotheses

    except Exception as e:
        print(f"  [ERROR] Valid batch ({domain}): {e}")
        return []


# ── Generate flawed hypotheses ────────────────────────────────────────────

def generate_flawed_batch(domain: str, flaw_type: str, batch_size: int = 5) -> list:
    """Ask Claude to generate intentionally flawed hypotheses."""

    flaw_descriptions = {
        "unfalsifiable":      "cannot be tested or disproven no matter what evidence is found",
        "too_vague":          "so vague that no specific experiment could confirm or deny them",
        "circular_reasoning": "where the conclusion is hidden in the premise",
        "no_mechanism":       "that state an outcome without explaining any mechanism for how it happens",
        "already_proven":     "that restate well-known established facts as if they were novel discoveries",
        "untestable_scope":   "that would require impossible experiments or infinite resources to test",
    }

    prompt = f"""Generate {batch_size} research hypotheses in {domain} that are 
intentionally FLAWED because they are {flaw_descriptions[flaw_type]}.

They should LOOK like real hypotheses but contain this specific flaw.
Between 1-3 sentences each.

Return ONLY a JSON array, no markdown:
[
  {{"hypothesis": "...", "testability": 0.0-1.0}},
  ...
]

testability = how easily this could ACTUALLY be tested (should be low for these flawed ones)"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        hypotheses = json.loads(text)

        for h in hypotheses:
            h['label'] = 0
            h['flaw_type'] = flaw_type
            h['domain'] = domain

        return hypotheses

    except Exception as e:
        print(f"  [ERROR] Flawed batch ({domain}/{flaw_type}): {e}")
        return []


# ── Main generation loop ──────────────────────────────────────────────────

def generate_full_dataset(target_valid: int = 15500, target_flawed: int = 15500):
    """
    Generate balanced dataset: ~50% valid, ~50% flawed.
    
    Valid:  15,500 = 20 domains × 775 each (78 batches of 10)
    Flawed: 15,500 = 20 domains × 6 flaw types × ~130 each (26 batches of 5)
    
    Total: ~31,000 examples
    """

    all_data = []
    output_path = os.path.join(os.path.dirname(__file__), 'data', 'hypotheses.csv')

    # ── Valid hypotheses ──────────────────────────────────────────────
    valid_per_domain = target_valid // len(DOMAINS)       # 775
    batches_per_domain = valid_per_domain // 10            # 77-78

    print(f"=== Generating Valid Hypotheses ===")
    print(f"Target: {target_valid} ({valid_per_domain} per domain, {batches_per_domain} batches)\n")

    for i, domain in enumerate(DOMAINS):
        domain_count = 0
        for batch_num in range(batches_per_domain):
            results = generate_valid_batch(domain, batch_size=10)
            all_data.extend(results)
            domain_count += len(results)

            if (batch_num + 1) % 10 == 0:
                print(f"  [{domain}] {domain_count} valid generated...")

            time.sleep(1)  # rate limit buffer

        print(f"  ✅ {domain}: {domain_count} valid hypotheses")

    # ── Flawed hypotheses ─────────────────────────────────────────────
    flawed_per_combo = target_flawed // (len(DOMAINS) * len(FLAW_TYPES))  # ~130
    batches_per_combo = flawed_per_combo // 5                               # ~26

    print(f"\n=== Generating Flawed Hypotheses ===")
    print(f"Target: {target_flawed} ({flawed_per_combo} per domain-flaw combo, {batches_per_combo} batches)\n")

    for domain in DOMAINS:
        domain_count = 0
        for flaw_type in FLAW_TYPES:
            for batch_num in range(batches_per_combo):
                results = generate_flawed_batch(domain, flaw_type, batch_size=5)
                all_data.extend(results)
                domain_count += len(results)
                time.sleep(1)

            print(f"  [{domain}] {flaw_type}: done")

        print(f"  ✅ {domain}: {domain_count} flawed hypotheses")

    # ── Shuffle and save ──────────────────────────────────────────────
    random.shuffle(all_data)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['hypothesis', 'label', 'testability', 'flaw_type', 'domain'])
        writer.writeheader()
        writer.writerows(all_data)

    valid_count  = sum(1 for d in all_data if d['label'] == 1)
    flawed_count = sum(1 for d in all_data if d['label'] == 0)

    print(f"\n=== Dataset Complete ===")
    print(f"Total:   {len(all_data)}")
    print(f"Valid:   {valid_count}")
    print(f"Flawed:  {flawed_count}")
    print(f"Saved:   {output_path}")

    return all_data


if __name__ == '__main__':
    generate_full_dataset(target_valid=1550, target_flawed=1550)