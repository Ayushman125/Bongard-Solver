# Debug script to find matching shape definitions in TSV
import csv
import os

tsv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Bongard-LOGO', 'Bongard-LOGO', 'data', 'human_designed_shapes.tsv'))

def parse_base_actions(actions_str):
    actions = [a.strip() for a in actions_str.split(',')]
    lengths = []
    for a in actions:
        if a.startswith('line') or a.startswith('arc'):
            parts = a.split('_')
            if len(parts) > 1:
                try:
                    val = round(float(parts[1]), 1)
                    lengths.append(val)
                except Exception:
                    pass
    return lengths

# Target signature
input_lengths = [1.0, 0.6, 0.2, 1.0, 0.2, 0.2]

def is_match(tsv_lengths, input_lengths, tol=0.15):
    if len(tsv_lengths) != len(input_lengths):
        return False
    for tsv_len, inp_len in zip(tsv_lengths, input_lengths):
        if abs(tsv_len - inp_len) > tol:
            return False
    return True

with open(tsv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        raw_actions = row.get('set of base actions', '')
        tsv_lengths = parse_base_actions(raw_actions)
        if is_match(tsv_lengths, input_lengths):
            print('MATCH FOUND:')
            print('Shape function name:', row.get('shape function name'))
            print('Raw actions repr:', repr(raw_actions))
            print('Parsed lengths:', tsv_lengths)
            print('---')
        else:
            # For debugging, print close candidates
            if len(tsv_lengths) == len(input_lengths):
                print('Candidate:')
                print('Shape function name:', row.get('shape function name'))
                print('Raw actions repr:', repr(raw_actions))
                print('Parsed lengths:', tsv_lengths)
                print('---')
