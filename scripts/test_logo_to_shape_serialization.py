import json

class LineAction:
    def __init__(self, raw_command):
        self.raw_command = raw_command

class ArcAction:
    def __init__(self, raw_command):
        self.raw_command = raw_command

def ensure_all_strings(lst):
    if isinstance(lst, list):
        return [ensure_all_strings(x) for x in lst]
    if hasattr(lst, 'raw_command'):
        return str(lst.raw_command)
    return str(lst)

def safe_join(lst, sep=','):
    if isinstance(lst, list):
        safe_items = [getattr(x, 'raw_command', str(x)) if not isinstance(x, str) else x for x in lst]
        safe_items = [str(x) for x in safe_items]
        return sep.join(safe_items)
    return str(lst)

# Test data
actions = [LineAction("line_normal_1.000-0.500"), LineAction("line_normal_0.200-0.750"), ArcAction("arc_normal_0.500_0.750-1.000")]

# Direct join (should fail)
try:
    print('Direct join:')
    print(','.join(actions))
except Exception as e:
    print('Error:', e)

# safe_join (should succeed)
print('\nSafe join:')
print(safe_join(actions))

# ensure_all_strings (should succeed)
print('\nEnsure all strings:')
print(ensure_all_strings(actions))

# Direct json dump (should fail)
try:
    print('\nDirect json dump:')
    print(json.dumps(actions))
except Exception as e:
    print('Error:', e)

# json dump after ensure_all_strings (should succeed)
print('\nSafe json dump:')
print(json.dumps(ensure_all_strings(actions)))
