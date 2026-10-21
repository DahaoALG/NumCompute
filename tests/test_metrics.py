"""Unit tests for :mod:`numcompute.metrics`: classification, regression and ROC metrics."""

import numpy as np
import pytest

from numcompute.metrics import accuracy, auc, confusion_matrix, f1, mse, precision, recall, roc_curve

# Binary example: tp=2, fp=1, fn=1, tn=1 for the positive class 1
BIN_TRUE = [0, 1, 1, 0, 1]
BIN_PRED = [0, 1, 0, 1, 1]

# Three classes with unequal support (3, 1, 2), worked out by hand:
#   confusion matrix [[2, 1, 0], [0, 1, 0], [1, 0, 1]]
#   tp = [2, 1, 1], fp = [1, 1, 0], fn = [1, 0, 1]
MULTI_TRUE = [0, 0, 0, 1, 2, 2]
MULTI_PRED = [0, 0, 1, 1, 2, 0]


@pytest.fixture
def random_labels():
    """1,000 random true/predicted labels from 4 classes, fixed seed."""
    rng = np.random.default_rng(7)

    return rng.integers(0, 4, size=1000), rng.integers(0, 4, size=1000)


# ---- accuracy ----

def test_accuracy_perfect_wrong_and_partial():
    """1.0 when all match, 0.0 when none do, the matching fraction otherwise."""
    assert accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert accuracy([1, 2, 3], [3, 1, 2]) == 0.0
    assert accuracy(MULTI_TRUE, MULTI_PRED) == pytest.approx(4 / 6)
    assert isinstance(accuracy(MULTI_TRUE, MULTI_PRED), float)


def test_accuracy_string_and_bool_labels():
    """Labels do not need to be numbers."""
    assert accuracy(["cat", "dog", "dog"], ["cat", "cat", "dog"]) == pytest.approx(2 / 3)
    assert accuracy(np.array([True, False]), np.array([True, True])) == 0.5


def test_accuracy_rejects_bad_inputs():
    """Different lengths, 2-D, empty input and NaN raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        accuracy([0, 1, 1], [0, 1])

    with pytest.raises(ValueError, match="1-D"):
        accuracy([[0], [1]], [[0], [1]])

    with pytest.raises(ValueError, match="empty"):
        accuracy([], [])

    with pytest.raises(ValueError, match="NaN"):
        accuracy([0.0, np.nan], [0.0, 1.0])


def test_accuracy_rejects_incompatible_and_unsupported_labels():
    """Numbers never match text, and object labels are not supported: TypeError, not a silent 0.0."""
    with pytest.raises(TypeError, match="incompatible"):
        accuracy([0, 1], ["0", "1"])

    with pytest.raises(TypeError, match="unsupported"):
        accuracy([None, 1], [0, 1])

    assert accuracy([0.0, 1.0], [0, 1]) == 1.0  # int and float labels compare by value


# ---- confusion_matrix ----

def test_confusion_matrix_hand_computed():
    """Rows are true labels, columns predicted labels, counts are ints."""
    cm = confusion_matrix(MULTI_TRUE, MULTI_PRED)

    np.testing.assert_array_equal(cm, [[2, 1, 0], [0, 1, 0], [1, 0, 1]])
    assert cm.dtype.kind == "i"


def test_confusion_matrix_labels_order_and_unseen_label():
    """``labels`` sets the order and can add a label with no samples."""
    cm = confusion_matrix(MULTI_TRUE, MULTI_PRED, labels=[2, 1, 0, 3])

    np.testing.assert_array_equal(cm, [[1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]])

    with pytest.raises(ValueError, match="not in labels"):
        confusion_matrix(MULTI_TRUE, MULTI_PRED, labels=[0, 1])

    with pytest.raises(ValueError, match="duplicates"):
        confusion_matrix(MULTI_TRUE, MULTI_PRED, labels=[0, 1, 2, 2])


def test_confusion_matrix_rejects_bad_labels():
    """``labels`` gets the same checks as the data, and must be the same type of label."""
    with pytest.raises(TypeError, match="incompatible"):
        confusion_matrix(MULTI_TRUE, MULTI_PRED, labels=["0", "1", "2"])

    with pytest.raises(TypeError, match="incompatible"):
        confusion_matrix([0, 1], ["0", "1"])

    with pytest.raises(ValueError, match="NaN"):
        confusion_matrix([0.0, 1.0], [0.0, 1.0], labels=[0.0, 1.0, np.nan])

    with pytest.raises(ValueError, match="1-D"):
        confusion_matrix(MULTI_TRUE, MULTI_PRED, labels=[[0, 1, 2]])


def test_confusion_matrix_string_labels_sorted():
    """String labels are sorted alphabetically by default."""
    cm = confusion_matrix(["dog", "cat", "dog", "bird"], ["dog", "dog", "dog", "bird"])

    # order: bird, cat, dog
    np.testing.assert_array_equal(cm, [[1, 0, 0], [0, 0, 1], [0, 0, 2]])


def test_confusion_matrix_matches_python_loop(random_labels):
    """The vectorised counts equal a plain loop over the samples."""
    y_true, y_pred = random_labels

    expected = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):  # reference implementation
        expected[t, p] += 1

    cm = confusion_matrix(y_true, y_pred)

    np.testing.assert_array_equal(cm, expected)
    assert cm.sum() == y_true.size
    assert np.trace(cm) == pytest.approx(accuracy(y_true, y_pred) * y_true.size)


# ---- precision ----

def test_precision_binary_by_hand():
    """tp / (tp + fp) for the chosen positive class."""
    assert precision(BIN_TRUE, BIN_PRED) == pytest.approx(2 / 3)
    assert precision(BIN_TRUE, BIN_PRED, pos_label=0) == pytest.approx(1 / 2)
    assert precision(["spam", "ham", "spam"], ["spam", "spam", "ham"], pos_label="spam") == 0.5


def test_precision_every_average_on_three_classes():
    """Per-class values and their macro, micro and weighted means."""
    per_class = precision(MULTI_TRUE, MULTI_PRED, average=None)

    np.testing.assert_allclose(per_class, [2 / 3, 1 / 2, 1])
    assert precision(MULTI_TRUE, MULTI_PRED, average="macro") == pytest.approx(per_class.mean())
    assert precision(MULTI_TRUE, MULTI_PRED, average="micro") == pytest.approx(4 / 6)
    assert precision(MULTI_TRUE, MULTI_PRED, average="weighted") == pytest.approx((3 * 2 / 3 + 1 * 1 / 2 + 2 * 1) / 6)


def test_precision_zero_division_when_nothing_predicted_positive():
    """No positive prediction gives zero_division instead of NaN."""
    y_true, y_pred = [0, 1, 1], [0, 0, 0]

    assert precision(y_true, y_pred) == 0.0
    assert precision(y_true, y_pred, zero_division=1.0) == 1.0
    assert not np.isnan(precision(y_true, y_pred, average=None)).any()


def test_precision_rejects_bad_average_and_pos_label():
    """Binary averaging needs two classes and a pos_label that is one of them."""
    with pytest.raises(ValueError, match="binary"):
        precision(MULTI_TRUE, MULTI_PRED)

    with pytest.raises(ValueError, match="pos_label"):
        precision(BIN_TRUE, BIN_PRED, pos_label=2)  # same type, but not one of the labels

    with pytest.raises(TypeError, match="pos_label"):
        precision(["spam", "ham"], ["spam", "spam"])  # default pos_label=1 is a number, labels are text

    with pytest.raises(ValueError, match="average"):
        precision(BIN_TRUE, BIN_PRED, average="mean")

    with pytest.raises(TypeError, match="zero_division"):
        precision(BIN_TRUE, BIN_PRED, zero_division="none")


def test_precision_rejects_bad_pos_label_and_zero_division():
    """pos_label must be one label of the same type; zero_division a real number."""
    with pytest.raises(TypeError, match="pos_label"):
        precision(["0", "1"], ["0", "1"], pos_label=1)  # would otherwise silently match "1"

    with pytest.raises(TypeError, match="single label"):
        precision(BIN_TRUE, BIN_PRED, pos_label=[1])

    with pytest.raises(TypeError, match="unsupported"):
        precision(BIN_TRUE, BIN_PRED, pos_label=None)

    with pytest.raises(TypeError, match="zero_division"):
        precision(BIN_TRUE, BIN_PRED, zero_division="0.5")  # text is not converted

    with pytest.raises(TypeError, match="zero_division"):
        precision(BIN_TRUE, BIN_PRED, zero_division=True)

    with pytest.raises(ValueError, match="NaN"):
        precision(BIN_TRUE, BIN_PRED, zero_division=np.nan)

    assert precision(BIN_TRUE, BIN_PRED, pos_label=np.int64(1)) == pytest.approx(2 / 3)  # NumPy scalars work


# ---- recall ----

def test_recall_binary_by_hand():
    """tp / (tp + fn) for the chosen positive class."""
    assert recall(BIN_TRUE, BIN_PRED) == pytest.approx(2 / 3)
    assert recall(BIN_TRUE, BIN_PRED, pos_label=0) == pytest.approx(1 / 2)


def test_recall_every_average_on_three_classes(random_labels):
    """Per-class values; weighted recall always equals accuracy."""
    np.testing.assert_allclose(recall(MULTI_TRUE, MULTI_PRED, average=None), [2 / 3, 1, 1 / 2])
    assert recall(MULTI_TRUE, MULTI_PRED, average="macro") == pytest.approx((2 / 3 + 1 + 1 / 2) / 3)
    assert recall(MULTI_TRUE, MULTI_PRED, average="weighted") == pytest.approx(accuracy(MULTI_TRUE, MULTI_PRED))

    y_true, y_pred = random_labels

    assert recall(y_true, y_pred, average="weighted") == pytest.approx(accuracy(y_true, y_pred))


def test_recall_zero_division_when_no_true_positive_class():
    """No sample of the positive class gives zero_division instead of NaN."""
    assert recall([0, 0, 0], [0, 1, 0]) == 0.0
    assert recall([0, 0, 0], [0, 1, 0], zero_division=1.0) == 1.0


def test_recall_single_class_input():
    """pos_label may be absent from the data: its counts are just 0."""
    assert recall([0, 0, 0], [0, 0, 0]) == 0.0
    assert recall([0, 0, 0], [0, 0, 0], pos_label=0) == 1.0


# ---- f1 ----

def test_f1_is_harmonic_mean_of_precision_and_recall(random_labels):
    """2 * p * r / (p + r), checked per class on random data."""
    y_true, y_pred = random_labels

    p = precision(y_true, y_pred, average=None)
    r = recall(y_true, y_pred, average=None)

    np.testing.assert_allclose(f1(y_true, y_pred, average=None), 2 * p * r / (p + r))
    assert f1(BIN_TRUE, BIN_PRED) == pytest.approx(2 / 3)


def test_f1_micro_equals_accuracy(random_labels):
    """With every class counted, micro F1 is the accuracy."""
    y_true, y_pred = random_labels

    assert f1(y_true, y_pred, average="micro") == pytest.approx(accuracy(y_true, y_pred))
    assert f1(MULTI_TRUE, MULTI_PRED, average="micro") == pytest.approx(4 / 6)


def test_f1_every_average_on_three_classes():
    """Per-class 2tp / (2tp + fp + fn) and the averages."""
    np.testing.assert_allclose(f1(MULTI_TRUE, MULTI_PRED, average=None), [4 / 6, 2 / 3, 2 / 3])
    assert f1(MULTI_TRUE, MULTI_PRED, average="macro") == pytest.approx(2 / 3)
    assert f1(MULTI_TRUE, MULTI_PRED, average="weighted") == pytest.approx(2 / 3)


def test_f1_all_wrong_and_zero_division():
    """All wrong gives 0; a class that never appears gives zero_division."""
    assert f1([0, 1, 0, 1], [1, 0, 1, 0]) == 0.0
    assert f1([0, 0], [0, 0]) == 0.0
    assert f1([0, 0], [0, 0], zero_division=1.0) == 1.0


# ---- mse ----

def test_mse_known_values():
    """Mean of squared differences over all samples."""
    assert mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)
    assert mse([1.5, -2.0], [1.5, -2.0]) == 0.0
    assert isinstance(mse([1, 2], [1, 3]), float)

    big = np.array([30000], dtype=np.int16)

    assert mse(big, -big) == 60000.0**2  # converted to float first: no int16 wrap-around


def test_mse_rejects_bad_length_2d_nan_empty_and_text():
    """Different lengths, 2-D, NaN, empty and non-numeric input are rejected."""
    with pytest.raises(ValueError, match="same length"):
        mse([1, 2, 3], [1, 2])

    with pytest.raises(ValueError, match="1-D"):
        mse([[1, 2], [3, 4]], [[1, 2], [3, 4]])

    with pytest.raises(ValueError, match="NaN"):
        mse([1.0, np.nan], [1.0, 2.0])

    with pytest.raises(ValueError, match="empty"):
        mse([], [])

    with pytest.raises(TypeError, match="numeric"):
        mse(["a", "b"], [1, 2])

    with pytest.raises(TypeError, match="numeric"):
        mse(["1", "2"], [1, 2])  # numbers written as text are not converted


def test_mse_non_contiguous_input():
    """Strided views give the same result as contiguous copies."""
    rng = np.random.default_rng(3)
    a = rng.normal(size=200)
    b = rng.normal(size=200)

    assert mse(a[::2], b[::2]) == pytest.approx(mse(a[::2].copy(), b[::2].copy()))
    assert mse(a[::-1], b[::-1]) == pytest.approx(mse(a, b))


# ---- roc_curve ----

def test_roc_curve_known_example():
    """Rates and thresholds for a small hand-checked example."""
    fpr, tpr, thresholds = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])

    np.testing.assert_allclose(fpr, [0, 0, 0.5, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 0.5, 1, 1])
    np.testing.assert_array_equal(thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])


def test_roc_curve_tied_scores_form_one_point():
    """Samples with the same score cross the threshold together."""
    fpr, tpr, thresholds = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.9])

    np.testing.assert_array_equal(thresholds, [np.inf, 0.9, 0.5])
    np.testing.assert_allclose(fpr, [0, 0, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 1])
    assert auc(fpr, tpr) == pytest.approx(0.75)  # a positive-negative tie counts as half


def test_roc_curve_rejects_one_class_many_classes_and_nan():
    """Both classes are needed, only two are allowed, scores cannot be NaN."""
    with pytest.raises(ValueError, match="positive and negative"):
        roc_curve([1, 1, 1], [0.2, 0.5, 0.9])

    with pytest.raises(ValueError, match="binary"):
        roc_curve([0, 1, 2], [0.2, 0.5, 0.9])

    with pytest.raises(ValueError, match="NaN"):
        roc_curve([0, 1], [0.2, np.nan])

    with pytest.raises(ValueError, match="same length"):
        roc_curve([0, 1, 1], [0.2, 0.5])


def test_roc_curve_string_labels_and_bad_pos_label():
    """Text labels work with a text pos_label; a mismatched pos_label or NaN label raises."""
    fpr, tpr, _ = roc_curve(["no", "no", "yes", "yes"], [0.1, 0.2, 0.8, 0.9], pos_label="yes")

    np.testing.assert_allclose(fpr, [0, 0, 0, 0.5, 1])
    np.testing.assert_allclose(tpr, [0, 0.5, 1, 1, 1])

    with pytest.raises(TypeError, match="pos_label"):
        roc_curve(["no", "yes"], [0.2, 0.9])  # default pos_label=1 is a number

    with pytest.raises(TypeError, match="single label"):
        roc_curve([0, 1], [0.2, 0.9], pos_label=[1])

    with pytest.raises(ValueError, match="y_true contains NaN"):
        roc_curve([0.0, np.nan], [0.2, 0.9])

    with pytest.raises(TypeError, match="numeric"):
        roc_curve([0, 1], ["0.2", "0.9"])


# ---- auc ----

def test_auc_known_areas_in_either_direction():
    """Trapezoid areas, with the same result for decreasing x."""
    assert auc([0, 1], [0, 1]) == 0.5
    assert auc([0, 0.5, 1], [1, 1, 1]) == 1.0
    assert auc([1, 0.5, 0], [1, 0.5, 0]) == 0.5
    assert auc([0, 0, 1], [0, 1, 1]) == 1.0  # repeated x (a vertical step) is allowed


def test_auc_rejects_non_monotonic_short_and_mismatched():
    """x must be monotonic, with at least 2 points matching y."""
    with pytest.raises(ValueError, match="monotonic"):
        auc([0, 1, 0.5], [0, 1, 1])

    with pytest.raises(ValueError, match="at least 2"):
        auc([0], [1])

    with pytest.raises(ValueError, match="same length"):
        auc([0, 1, 2], [0, 1])


def test_auc_rejects_text_2d_and_nan():
    """Coordinates must be 1-D numbers without NaN."""
    with pytest.raises(TypeError, match="numeric"):
        auc(["0", "1"], [0, 1])

    with pytest.raises(ValueError, match="1-D"):
        auc([[0, 1]], [[0, 1]])

    with pytest.raises(ValueError, match="NaN"):
        auc([0, 1], [0, np.nan])
