"""FCI relabelling rules for the IIT Indore dataset (docs/FCI_REQUIREMENTS_TRACE.md section 4)."""
import pytest

from training.relabel_iit import fci_label, load_rules

LM, THR = load_rules()
WHOLE = 5.24  # BPT5204 whole-kernel length (mm)


@pytest.mark.parametrize("src,length,frac,expected", [
    ("Healthy", 5.2, None, "sound"),
    ("Healthy", 3.5, None, "broken"),          # healthy but < 3/4 -> broken
    ("Broken", 3.0, None, "broken"),
    ("Broken", 5.1, None, "sound"),            # source says broken, FCI length rule says whole
    ("Broken", 1.2, None, "small_broken"),     # < 1/4
    ("Full chalky", 5.0, None, "chalky"),
    ("Full chalky", 3.0, None, "broken"),      # broken outranks chalky in picking order
    ("Partial chalky", 5.0, 0.62, "chalky"),   # >= 1/2 milky
    ("Partial chalky", 5.0, 0.30, "sound"),    # < 1/2 milky -> not FCI chalky
    ("Partial chalky", 5.0, None, "chalky_unmeasured"),
    ("Peck damage", 2.0, None, "damaged"),     # damaged outranks broken
    ("Discolored", 5.2, None, "damaged"),      # default pending FCI Q1
    ("PB1121", 8.6, None, "sound"),
])
def test_rules(src, length, frac, expected):
    assert fci_label(src, length, WHOLE, frac, LM, THR)[0] == expected


def test_discolored_flagged_pending():
    assert "Q1" in LM["labels"]["Discolored"]["pending"]


def test_unknown_label():
    assert fci_label("Mystery", 5, WHOLE, None, LM, THR)[0] == "unmapped"
