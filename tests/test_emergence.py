import unittest
from src.Derive_labels.emergence import EmergenceDetector, AbstractionHierarchy, ConceptMemoryBank

class TestEmergence(unittest.TestCase):
    def test_mine_stroke_patterns(self):
        motifs = EmergenceDetector.mine_stroke_patterns(['line_normal', 'arc_square'])
        self.assertIsInstance(motifs, list)
    def test_detect_emergent_concepts(self):
        emergent = EmergenceDetector.detect_emergent_concepts(['motif1'], set())
        self.assertIsInstance(emergent, list)
    def test_abstract(self):
        abstracted = AbstractionHierarchy.abstract(['emergent_concept1'])
        self.assertIsInstance(abstracted, list)
    def test_memory_bank(self):
        ConceptMemoryBank.initialize()
        ConceptMemoryBank.integrate(['conceptA'])
        mem = ConceptMemoryBank.load()
        self.assertIn('conceptA', mem)

if __name__ == "__main__":
    unittest.main()
