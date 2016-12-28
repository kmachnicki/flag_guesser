from enum import Enum


class WindowType(Enum):
    Boolean = 0
    Enumeration = 1
    Numeric = 2

TRAIN_SET_PATH = "data_sets/flags_clean.csv"
ALGORITHM = "l-bfgs"
MAX_ITER = 500
ALPHA = 0.88
HIDDEN_LAYER_SIZE = 8
RANDOM_STATE = 1

COLORS = [
    ("red", 1),
    ("blue", 2),
    ("green", 3),
    ("yellow (gold)", 4),
    ("white", 5),
    ("black", 7),
    ("orange (brown)", 8),
    ("other", 9),
]

# questions order based on a features selection
QUESTIONS_ORDER = (15, 1, 0, 10, 22, 21, 6, 5, 4, 12, 11, 7, 8, 14, 18, 3, 19, 2, 17, 9, 13, 20, 16)

QUESTIONS = [
    ("Number of vertical bars in the flag:", WindowType.Numeric),
    ("Number of horizontal stripes in the flag:", WindowType.Numeric),
    ("Number of different colours in the flag:", WindowType.Numeric),
    ("Is red colour present in the flag?", WindowType.Boolean),
    ("Is green colour present in the flag?", WindowType.Boolean),
    ("Is blue colour present in the flag?", WindowType.Boolean),
    ("Is gold or yellow colour present in the flag?", WindowType.Boolean),
    ("Is white colour present in the flag?", WindowType.Boolean),
    ("Is black colour present in the flag?", WindowType.Boolean),
    ("Is orange or brown colour present in the flag?", WindowType.Boolean),
    ("What is the predominant colour in the flag?\n"
     "(tie-breaks decided by taking the topmost hue,\n"
     "if that fails then the most central hue,\n"
     "and if that fails the leftmost hue)", WindowType.Enumeration),
    ("Number of circles in the flag:", WindowType.Numeric),
    ("Number of (upright) crosses:", WindowType.Numeric),
    ("Number of diagonal crosses:", WindowType.Numeric),
    ("Number of quartered sections:", WindowType.Numeric),
    ("Number of sun or star symbols:", WindowType.Numeric),
    ("Is crescent moon symbol present in the flag?", WindowType.Boolean),
    ("Are any triangles present in the flag?", WindowType.Boolean),
    ("Is an inanimate image (e.g., a boat or an emblem) present in the flag? ", WindowType.Boolean),
    ("Is an animate image (e.g., an eagle, a tree, a human hand) present in the flag?", WindowType.Boolean),
    ("Are any letters or writing (e.g., a motto or slogan) present in the flag?", WindowType.Boolean),
    ("Colour in the top-left corner\n(moving right to decide tie-breaks):", WindowType.Enumeration),
    ("Colour in the bottom-right corner\n(moving left to decide tie-breaks):", WindowType.Enumeration),
]
