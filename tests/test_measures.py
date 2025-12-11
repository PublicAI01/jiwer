import functools
import unittest

import jiwer


def all_m(wer, mer, wil):
    return {
        "wer": wer,
        "mer": mer,
        "wip": 1 - wil,
        "wil": wil,
    }


def to_measure_dict(x: jiwer.WordOutput):
    return {"wer": x.wer, "mer": x.mer, "wip": x.wip, "wil": x.wil}


def assert_dict_almost_equal(
    test_case: unittest.TestCase, a, b, places=None, msg=None, delta=None
):
    test_case.assertIsInstance(a, dict)
    test_case.assertIsInstance(b, dict)
    test_case.assertEqual(set(a.keys()), set(b.keys()))

    for k in a.keys():
        test_case.assertAlmostEqual(a[k], b[k], places=places, msg=msg, delta=delta)


class TestMeasuresContiguousSentencesTransform(unittest.TestCase):
    def test_input_ref_string_hyp_string(self):
        cases = [
            ("This is a test", "This is a test", all_m(0, 0, 0)),
            ("This is a test", "", all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            ("This is a test", "This test", all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_ref_string_hyp_list(self):
        cases = [
            ("This is a test", ["This is a test"], all_m(0, 0, 0)),
            ("This is a test", [""], all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            ("This is a test", ["This test"], all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_ref_list_hyp_string(self):
        cases = [
            (["This is a test"], "This is a test", all_m(0, 0, 0)),
            (["This is a test"], "", all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            (["This is a test"], "This test", all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_ref_list_hyp_list(self):
        cases = [
            (["This is a test"], ["This is a test"], all_m(0, 0, 0)),
            (["This is a test"], [""], all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            (["This is a test"], ["This test"], all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_different_sentence_length_equal_type(self):
        cases = [
            (
                ["hello", "this", "sentence", "is fractured"],
                ["this sentence"],
                all_m(0.24, 0.6, 0.6),  # Weighted: 3 dels * 0.4 / 5 = 1.2/5 = 0.24
            ),
            (
                "i am a short ground truth",
                "i am a considerably longer and very much incorrect hypothesis",
                all_m(11 / 30, 0.7, 0.85),  # Weighted: (3*0.2 + 4*0.4)/6 = 2.2/6 = 11/30 ≈ 0.3667
            ),
        ]

        self._apply_test_on(cases)

    def test_different_sentence_length_unequal_type(self):
        reference = [
            "i like monthy python",
            "what do you mean african or european swallow",
        ]
        hypothesis = ["i like", "python", "what you mean", "or swallow"]
        x = jiwer.process_words(
            reference,
            hypothesis,
            reference_transform=jiwer.transformations.wer_contiguous,
            hypothesis_transform=jiwer.transformations.wer_contiguous,
        )
        x_dict = to_measure_dict(x)

        # is equivalent to

        reference = "i like monthy python what do you mean african or european swallow"
        hypothesis = "i like python what you mean or swallow"
        y = jiwer.process_words(
            reference,
            hypothesis,
            reference_transform=jiwer.transformations.wer_contiguous,
            hypothesis_transform=jiwer.transformations.wer_contiguous,
        )
        y_dict = to_measure_dict(y)

        assert_dict_almost_equal(self, x_dict, y_dict, delta=1e-9)

    def test_known_values(self):
        # Taken from the "From WER and RIL to MER and WIL" paper, for link see README.md
        # NOTE: WER values updated for weighted calculation (S=0.2, D=0.4, I=0.4)
        cases = [
            (
                "X",
                "X",
                all_m(0, 0, 0),  # No errors
            ),
            (
                "X",
                "X X Y Y",
                all_m(1.2, 0.75, 0.75),  # 3 insertions: (3*0.4)/1 = 1.2
            ),
            (
                "X Y X",
                "X Z",
                all_m(0.2, 2 / 3, 5 / 6),  # 1 sub, 1 del: (1*0.2 + 1*0.4)/3 = 0.6/3 = 0.2
            ),
            (
                "X",
                "Y",
                all_m(0.2, 1, 1),  # 1 substitution: (1*0.2)/1 = 0.2
            ),
            (
                "X",
                "Y Z",
                all_m(0.6, 1, 1),  # 1 sub, 1 ins: (1*0.2 + 1*0.4)/1 = 0.6/1 = 0.6
            ),
        ]

        self._apply_test_on(cases)

    def test_permutations_variance(self):
        cases = [
            (
                ["i", "am i good"],
                ["i am", "i good"],
                all_m(0.0, 0.0, 0),
            ),
            (
                ["am i good", "i"],
                [
                    "i good",
                    "i am",
                ],
                all_m(0.2, 0.4, 7 / 16),  # Weighted: 1 del, 1 ins: (1*0.4 + 1*0.4)/4 = 0.8/4 = 0.2
            ),
        ]

        self._apply_test_on(cases)

    def _apply_test_on(self, cases):
        for ref, hyp, correct_measures in cases:
            output = jiwer.process_words(
                reference=ref,
                hypothesis=hyp,
                reference_transform=jiwer.transformations.wer_contiguous,
                hypothesis_transform=jiwer.transformations.wer_contiguous,
            )
            output_dict = to_measure_dict(output)

            assert_dict_almost_equal(self, output_dict, correct_measures, delta=1e-9)


class TestMeasuresDefaultTransform(unittest.TestCase):
    def test_input_gt_string_h_string(self):
        cases = [
            ("This is a test", "This is a test", all_m(0, 0, 0)),
            ("This is a test", "", all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            ("This is a test", "This test", all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_gt_string_h_list(self):
        cases = [
            ("This is a test", ["This is a test"], all_m(0, 0, 0)),
            ("This is a test", [""], all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            ("This is a test", ["This test"], all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_gt_list_h_string(self):
        cases = [
            (["This is a test"], "This is a test", all_m(0, 0, 0)),
            (["This is a test"], "", all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            (["This is a test"], "This test", all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_input_gt_list_h_list(self):
        cases = [
            (["This is a test"], ["This is a test"], all_m(0, 0, 0)),
            (["This is a test"], [""], all_m(0.4, 1, 1)),  # Weighted: 4 deletions * 0.4 / 4 = 0.4
            (["This is a test"], ["This test"], all_m(0.2, 0.5, 0.5)),  # Weighted: 2 deletions * 0.4 / 4 = 0.2
        ]

        self._apply_test_on(cases)

    def test_fail_on_different_sentence_length(self):
        for method in [
            jiwer.process_words,
            jiwer.wer,
            jiwer.wil,
            jiwer.wip,
            jiwer.mer,
        ]:
            self.assertRaises(
                ValueError,
                functools.partial(
                    method,
                    ["hello", "this", "sentence", "is fractured"],
                    ["this sentence"],
                ),
            )

    def test_known_values(self):
        # Taken from the "From WER and RIL to MER and WIL" paper, for link see README.md
        # NOTE: WER values updated for weighted calculation (S=0.2, D=0.4, I=0.4)
        cases = [
            (
                "X",
                "X",
                all_m(0, 0, 0),  # No errors
            ),
            (
                "X",
                "X X Y Y",
                all_m(1.2, 0.75, 0.75),  # 3 insertions: (3*0.4)/1 = 1.2
            ),
            (
                "X Y X",
                "X Z",
                all_m(0.2, 2 / 3, 5 / 6),  # 1 sub, 1 del: (1*0.2 + 1*0.4)/3 = 0.6/3 = 0.2
            ),
            (
                "X",
                "Y",
                all_m(0.2, 1, 1),  # 1 substitution: (1*0.2)/1 = 0.2
            ),
            (
                "X",
                "Y Z",
                all_m(0.6, 1, 1),  # 1 sub, 1 ins: (1*0.2 + 1*0.4)/1 = 0.6/1 = 0.6
            ),
        ]

        self._apply_test_on(cases)

    def test_permutations_invariance(self):
        cases = [
            (
                ["i", "am i good"],
                ["i am", "i good"],
                all_m(0.2, 0.4, 7 / 16),  # Weighted: 1 del, 1 ins: (1*0.4 + 1*0.4)/4 = 0.8/4 = 0.2
            ),
            (
                ["am i good", "i"],
                [
                    "i good",
                    "i am",
                ],
                all_m(0.2, 0.4, 7 / 16),  # Weighted: 1 del, 1 ins: (1*0.4 + 1*0.4)/4 = 0.8/4 = 0.2
            ),
        ]

        self._apply_test_on(cases)

    def _apply_test_on(self, cases):
        for ref, hyp, correct_measures in cases:
            output = jiwer.process_words(reference=ref, hypothesis=hyp)
            output_dict = to_measure_dict(output)

            assert_dict_almost_equal(self, output_dict, correct_measures, delta=1e-9)
