
from src.Derive_labels.file_io import FileIO

class TSVValidator:
    def __init__(self, tsv_path):
        self.rows = FileIO._load_tsv(tsv_path)
        # Build a mapping from problem_id to row for fast lookup
        self.row_map = {row['problem_id']: row for row in self.rows if 'problem_id' in row}

    def validate(self, pid, record):
        row = self.row_map.get(pid)
        if not row:
            return {'emergent_accuracy': None, 'meta_correct': None}
        # Map ground truth concepts if present
        gt_concepts = row.get('ground_truth_concepts')
        if isinstance(gt_concepts, str):
            import ast
            gt_concepts = ast.literal_eval(gt_concepts)
        emerg_acc = None
        if gt_concepts:
            emerg_acc = sum(record.get('emergent_concepts', {}).get(k)==v for k,v in gt_concepts.items())/len(gt_concepts)
        meta_ok = (record.get('meta_prob',0)>0.5)==bool(row.get('label', False))
        return {'emergent_accuracy': emerg_acc, 'meta_correct': meta_ok}
