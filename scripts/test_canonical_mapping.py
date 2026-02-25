import math
import numpy as np


import re
def parse_lengths(actions_str):
    # Robustly extract lengths for both line and arc actions
    actions = [a.strip() for a in actions_str.split(',')]
    lengths = []
    for a in actions:
        # line_X
        m = re.match(r'(?:line|arc)_([0-9.]+)', a)
        if m:
            val = float(m.group(1))
            lengths.append(val)
        else:
            # arc_X_Y (radius, angle)
            m2 = re.match(r'arc_([0-9.]+)_([0-9.]+)', a)
            if m2:
                val = float(m2.group(1))  # radius as length
                lengths.append(val)
    return lengths


def parse_tsv_signed_angles(angle_str):
    # TSV: 'L0--R138.18--L37.875--R137.121--L47.121--L126.87'
    seq = []
    for part in angle_str.split('--'):
        sign = +1 if part.startswith('L') else -1
        val = float(part.lstrip('LR'))
        seq.append((sign * val) % 360)
    return seq

def normalize_ratios(lengths):
    s = sum(lengths)
    return [l/s for l in lengths]

def all_cyclic_rotations(seq):
    n = len(seq)
    return [seq[i:] + seq[:i] for i in range(n)]


def match_lengths(r1, r2, tol=0.05):
    return all(abs(a-b) < tol for a, b in zip(r1, r2))

def normalized_to_degrees(t):
    return t * 360.0


def candidate_signed_sequences(frac_angles):
    # JSON fractions -> degrees
    deg = [f * 360.0 for f in frac_angles]
    pos = [round(a,3) for a in deg]
    neg = [round(-a % 360,3) for a in deg]
    return [pos, neg]



def match_angles_per_stroke(tsv_angles_str, frac_angles, tol=5.0):
    # Decode JSON fractions to signed degrees using Bongard normalization logic
    def decode_json_turn_angle(norm):
        if norm >= 0.5:
            angle = norm * 360.0 - 180.0
            sign = 'L'
        else:
            angle = 180.0 - norm * 360.0
            sign = 'R'
        return sign, angle
    decoded = [decode_json_turn_angle(f) for f in frac_angles]
    degs = [(a if s == 'L' else -a) % 360 for s, a in decoded]
    tsv = [float(a.lstrip('LR')) if a.startswith('L') else -float(a.lstrip('LR')) for a in tsv_angles_str.split('--')]
    # Print raw TSV and JSON angles side-by-side
    print('\n[DEEP DIAGNOSTIC] Raw TSV and JSON angles:')
    for i, (t, (s, a)) in enumerate(zip(tsv, decoded)):
        print(f"  Stroke {i}: TSV={round(t,2)}, JSON=({s},{round(a,2)}), Diff={round(abs(t-(a if s=='L' else -a)),2)}")

    def print_mismatch_summary(rot, tsv, label):
        print(f"\n[MISMATCH SUMMARY] {label}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            print(f"  Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, Diff={round(abs(d-t),2)}")

    found = False
    # Try all cyclic rotations
    for rot_idx, rot in enumerate(all_cyclic_rotations(degs)):
        good = True
        print(f"\n[DEBUG] Cyclic rotation {rot_idx}: {np.round(rot, 2)}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            print(f"  Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, |direct|={round(diff_direct,2)}, |complement|={round(diff_complement,2)}")
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"[DEBUG] Match found with cyclic rotation {rot_idx}")
            found = True
        else:
            print_mismatch_summary(rot, tsv, f'Cyclic rotation {rot_idx}')

    # Try offsetting JSON sequence by one (rotate left by one)
    offset_degs = degs[1:] + degs[:1]
    print('\n[DEEP DIAGNOSTIC] Trying offset (rotate left by one) JSON sequence:')
    for rot_idx, rot in enumerate(all_cyclic_rotations(offset_degs)):
        good = True
        print(f"  Offset Cyclic rotation {rot_idx}: {np.round(rot, 2)}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            print(f"    Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, |direct|={round(diff_direct,2)}, |complement|={round(diff_complement,2)}")
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"  [DEBUG] Match found with offset cyclic rotation {rot_idx}")
            found = True
        else:
            print_mismatch_summary(rot, tsv, f'Offset cyclic rotation {rot_idx}')

    # Try per-stroke sign flip (for each stroke, try both sign)
    print('\n[DEEP DIAGNOSTIC] Trying per-stroke sign flip:')
    for flip_mask in range(1, 2**len(degs)):
        flip_degs = [(d if (flip_mask >> i) & 1 == 0 else (-d)%360) for i, d in enumerate(degs)]
        good = True
        for i, (d, t) in enumerate(zip(flip_degs, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"  [DEBUG] Match found with per-stroke sign flip mask {bin(flip_mask)[2:].zfill(len(degs))}")
            found = True
        else:
            print_mismatch_summary(flip_degs, tsv, f'Per-stroke sign flip mask {bin(flip_mask)[2:].zfill(len(degs))}')

    # Try reversed stroke order
    rev_degs = list(reversed(degs))
    print('\n[DEEP DIAGNOSTIC] Trying reversed stroke order:')
    for rot_idx, rot in enumerate(all_cyclic_rotations(rev_degs)):
        good = True
        print(f"  Reversed Cyclic rotation {rot_idx}: {np.round(rot, 2)}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            print(f"    Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, |direct|={round(diff_direct,2)}, |complement|={round(diff_complement,2)}")
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"  [DEBUG] Match found with reversed cyclic rotation {rot_idx}")
            found = True
        else:
            print_mismatch_summary(rot, tsv, f'Reversed cyclic rotation {rot_idx}')

    # Try global sign flip
    flip_degs = [(-d)%360 for d in degs]
    print('\n[DEEP DIAGNOSTIC] Trying global sign flip:')
    for rot_idx, rot in enumerate(all_cyclic_rotations(flip_degs)):
        good = True
        print(f"  Sign-flip Cyclic rotation {rot_idx}: {np.round(rot, 2)}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            print(f"    Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, |direct|={round(diff_direct,2)}, |complement|={round(diff_complement,2)}")
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"  [DEBUG] Match found with sign-flip cyclic rotation {rot_idx}")
            found = True
        else:
            print_mismatch_summary(rot, tsv, f'Sign-flip cyclic rotation {rot_idx}')

    # Try reversed cyclic rotations of sign-flip
    rev_flip_degs = list(reversed(flip_degs))
    print('\n[DEEP DIAGNOSTIC] Trying reversed stroke order with sign flip:')
    for rot_idx, rot in enumerate(all_cyclic_rotations(rev_flip_degs)):
        good = True
        print(f"  Rev+Flip Cyclic rotation {rot_idx}: {np.round(rot, 2)}")
        for i, (d, t) in enumerate(zip(rot, tsv)):
            diff_direct = abs(d - t)
            diff_complement = abs(d - (360 - t))
            print(f"    Stroke {i}: JSON={round(d,2)}, TSV={round(t,2)}, |direct|={round(diff_direct,2)}, |complement|={round(diff_complement,2)}")
            if not (diff_direct < tol or diff_complement < tol):
                good = False
        if good:
            print(f"  [DEBUG] Match found with rev+flip cyclic rotation {rot_idx}")
            found = True
        else:
            print_mismatch_summary(rot, tsv, f'Rev+Flip cyclic rotation {rot_idx}')

    if found:
        print('[DEEP DIAGNOSTIC] At least one matching strategy succeeded!')
        return True
    print("[DEBUG] No match found in any cyclic rotation, offset, per-stroke sign flip, reversed, or sign-flip.")
    return False


def parse_action_lengths(action_cmds):
    # Robustly extract lengths/radii from action program commands
    lengths = []
    for cmd in action_cmds:
        # line_normal_0.657-0.500 or arc_normal_0.500_0.625-0.684
        m = re.match(r'(?:line|arc)_[^_]+_([0-9.]+)', cmd)
        if m:
            val = float(m.group(1))
            lengths.append(val)
        else:
            m2 = re.match(r'arc_[^_]+_([0-9.]+)_([0-9.]+)-([0-9.]+)', cmd)
            if m2:
                val = float(m2.group(1))  # radius
                lengths.append(val)
    return lengths

def test_mapping(tsv_row, action_cmds):
    # Only match on normalized length ratios (lines and arcs)
    tsv_lengths = parse_lengths(tsv_row['set of base actions'])
    act_lengths = parse_action_lengths(action_cmds)
    tsv_ratios = normalize_ratios(tsv_lengths)
    act_ratios = normalize_ratios(act_lengths)
    print('TSV lengths:', tsv_lengths)
    print('ACT lengths:', act_lengths)
    print('TSV ratios:', np.round(tsv_ratios, 3))
    print('ACT ratios:', np.round(act_ratios, 3))
    if len(tsv_ratios) != len(act_ratios):
        print('Length count mismatch!')
        return False
    def measure_ratio_similarity(r1, r2):
        diffs = [abs(a-b) for a, b in zip(r1, r2)]
        max_diff = max(diffs)
        mean_diff = np.mean(diffs)
        print(f"[RATIO DIAGNOSTIC] Max diff: {max_diff:.4f}, Mean diff: {mean_diff:.4f}")
        return max_diff, mean_diff
    measure_ratio_similarity(tsv_ratios, act_ratios)
    if not match_lengths(tsv_ratios, act_ratios):
        print('Length ratios do not match!')
        return False
    print('Mapping successful!')
    return True

if __name__ == '__main__':
    # Example TSV row and action commands
    tsv_row = {
        'shape function name': 'asymmetric_unbala_goldfish',
        'set of base actions': 'line_1.0, line_0.51, line_0.224, line_0.854, line_0.224, line_0.224',
        'turn angles': 'L0--R138.18--L37.875--R137.121--L47.121--L126.87'
    }
    action_cmds = [
        'line_normal_1.000-0.500',
        'line_normal_0.510-0.121',
        'line_normal_0.224-0.600',
        'line_normal_0.854-0.074',
        'line_normal_0.224-0.926',
        'line_normal_0.224-0.352',
    ]
    test_mapping(tsv_row, action_cmds)
