import unittest

import jiwer


class TestCERInputMethods(unittest.TestCase):
    def test_input_ref_string_hyp_string(self):
        cases = [
            ("This is a test", "This is a test", 0 / 14),  # No errors
            ("This is a test", "", 14 * 0.4 / 14),  # 14 deletions: (14*0.4)/14 = 0.4
            ("This is a test", "This test", 5 * 0.4 / 14),  # 5 deletions: (5*0.4)/14 = 0.1429
        ]

        self._apply_test_on(cases)

    def test_input_ref_string_hyp_list(self):
        cases = [
            ("This is a test", ["This is a test"], 0 / 14),  # No errors
            ("This is a test", [""], 14 * 0.4 / 14),  # 14 deletions: (14*0.4)/14 = 0.4
            ("This is a test", ["This test"], 5 * 0.4 / 14),  # 5 deletions: (5*0.4)/14 = 0.1429
        ]

        self._apply_test_on(cases)

    def test_input_ref_list_hyp_string(self):
        cases = [
            (["This is a test"], "This is a test", 0 / 14),  # No errors
            (["This is a test"], "", 14 * 0.4 / 14),  # 14 deletions: (14*0.4)/14 = 0.4
            (["This is a test"], "This test", 5 * 0.4 / 14),  # 5 deletions: (5*0.4)/14 = 0.1429
        ]

        self._apply_test_on(cases)

    def test_input_ref_list_hyp_list(self):
        cases = [
            (["This is a test"], ["This is a test"], 0 / 14),  # No errors
            (["This is a test"], [""], 14 * 0.4 / 14),  # 14 deletions: (14*0.4)/14 = 0.4
            (["This is a test"], ["This test"], 5 * 0.4 / 14),  # 5 deletions: (5*0.4)/14 = 0.1429
        ]

        self._apply_test_on(cases)

    def test_fail_on_different_sentence_length(self):
        def callback():
            jiwer.cer(["hello", "this", "sentence", "is fractured"], ["this sentence"])

        self.assertRaises(ValueError, callback)

    def test_known_values(self):
        # Taken from the "From WER and RIL to MER and WIL" paper, for link see README.md
        # NOTE: CER values updated for weighted calculation (S=0.2, D=0.4, I=0.4)
        cases = [
            (
                "X",
                "X",
                0,  # No errors
            ),
            (
                "X",
                "X X Y Y",
                2.4,  # 6 insertions: (6*0.4)/1 = 2.4
            ),
            (
                "X Y X",
                "X Z",
                0.2,  # 1 sub, 2 dels: (1*0.2 + 2*0.4)/5 = 1.0/5 = 0.2
            ),
            (
                "X",
                "Y",
                0.2,  # 1 substitution: (1*0.2)/1 = 0.2
            ),
            (
                "X",
                "Y Z",
                1.0,  # 1 sub, 2 ins: (1*0.2 + 2*0.4)/1 = 1.0
            ),
        ]

        self._apply_test_on(cases)

    def test_permutations_invariance(self):
        cases = [
            (
                ["i", "am i good"],
                ["i am", "i good"],
                0.24,  # 3 dels, 3 ins: (3*0.4 + 3*0.4)/10 = 2.4/10 = 0.24
            ),
            (
                ["am i good", "i"],
                [
                    "i good",
                    "i am",
                ],
                0.24,  # 3 dels, 3 ins: (3*0.4 + 3*0.4)/10 = 2.4/10 = 0.24
            ),
        ]

        self._apply_test_on(cases)

    def _apply_test_on(self, cases):
        for ref, hyp, correct_cer in cases:
            cer = jiwer.cer(reference=ref, hypothesis=hyp)

            self.assertTrue(isinstance(cer, float))
            if isinstance(cer, float):
                self.assertAlmostEqual(cer, correct_cer, delta=1e-9)
