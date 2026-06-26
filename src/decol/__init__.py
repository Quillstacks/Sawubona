"""Linguistic Decolonization: conditioned political bias of edge LLMs across South
Africa's languages.

Pipeline (per language):
    1. elicit   - administer Afrobarometer political items to edge models (Ollama)
    2. capability - score per-language capability (AfroBench + extension)
    3. divergence - compare model answer distribution to speakers' real attitudes
    4. analysis - correlate divergence with capability and model size
"""

__version__ = "0.1.0"
