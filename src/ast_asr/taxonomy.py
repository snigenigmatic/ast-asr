"""Explicit language-family taxonomy used by Svarah experiments."""

from __future__ import annotations

# The public Svarah accent table names these 19 primary languages. Linguistic
# classification yields three represented families. The dataset card's prose
# says "four language families", but neither its public schema nor accent table
# identifies a fourth. We therefore encode the auditable 19-language mapping
# and reject unknown labels instead of inventing a fourth group.
SVARAH_LANGUAGE_FAMILIES: dict[str, str] = {
    "Assamese": "Indo-Aryan",
    "Bengali": "Indo-Aryan",
    "Bodo": "Sino-Tibetan",
    "Dogri": "Indo-Aryan",
    "Gujarati": "Indo-Aryan",
    "Hindi": "Indo-Aryan",
    "Kannada": "Dravidian",
    "Kashmiri": "Indo-Aryan",
    "Konkani": "Indo-Aryan",
    "Maithili": "Indo-Aryan",
    "Malayalam": "Dravidian",
    "Marathi": "Indo-Aryan",
    "Nepali": "Indo-Aryan",
    "Odia": "Indo-Aryan",
    "Punjabi": "Indo-Aryan",
    "Sindhi": "Indo-Aryan",
    "Tamil": "Dravidian",
    "Telugu": "Dravidian",
    "Urdu": "Indo-Aryan",
}

TAXONOMY_RECONCILIATION = (
    "Svarah's public card says four language families, while its published "
    "19-language accent table resolves to Indo-Aryan, Dravidian, and "
    "Sino-Tibetan. This manifest uses that explicit three-family mapping and "
    "fails on any unseen primary language."
)
