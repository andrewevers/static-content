def add(a, b):
    """Return the sum of two numbers."""
    return a + b


def capitalize_word(word):
    """Return the word with its first letter capitalized."""
    return word.capitalize()


def count_occurrences(text, word):
    """Return the number of times word appears in text (case-insensitive)."""
    return text.lower().split().count(word.lower())
