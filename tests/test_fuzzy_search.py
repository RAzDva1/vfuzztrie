from vfuzztrie import prefix_search_builder, PrefixSearch
import pytest

TERMS = [
    ("эдисон", "123e4567-e89b-12d3-a456-426614174000", 10),
    ("эдесон", "123e4567-e89b-12d3-a456-426614174001", 5),
    ("эдиссон перец", "123e4567-e89b-12d3-a456-426614174002", 8),
    ("эдессон", "123e4567-e89b-12d3-a456-426614174003", 7),
    ("эдессо", "123e4567-e89b-12d3-a456-426614174004", 5),
]


@pytest.fixture(scope="module")
def prefix_search() -> PrefixSearch:
    return prefix_search_builder.from_nodes_video(TERMS)


def test_exact_match(prefix_search: PrefixSearch):
    assert prefix_search.fuzzy_match_video("эдисон", 0, 1)[0][1] == 1.0
    assert prefix_search.fuzzy_match_video("эдисон", 0, 1)[0][0] == "123e4567-e89b-12d3-a456-426614174000"


def test_fuzzy_match_distance_eq_1(prefix_search: PrefixSearch):
    assert set(res[0] for res in prefix_search.fuzzy_match_video("эдессон", 1, None)) == {'123e4567-e89b-12d3-a456-426614174001','123e4567-e89b-12d3-a456-426614174003', '123e4567-e89b-12d3-a456-426614174004' }


def test_fuzzy_match_distance_eq_1_with_limit(prefix_search: PrefixSearch):
    assert len(prefix_search.fuzzy_match_video("идисон", 1, 1)) == 1


def test_fuzzy_match_distance_eq_2(prefix_search: PrefixSearch):
    assert set(res[0] for res in prefix_search.fuzzy_match_video("эдисссонн перец", 2, None)) == {"123e4567-e89b-12d3-a456-426614174002"}


def test_min_score_default_filters_low_scores_positive(prefix_search: PrefixSearch):
    """Positive test: default min_score=0.01 filters out low-scoring results."""
    results_default = prefix_search.fuzzy_match_video("эд", 0, None)
    results_lower = prefix_search.fuzzy_match_video("эд", 0, None, None, 0.001)
    assert len(results_lower) >= len(results_default)


def test_min_score_zero_returns_all_positive(prefix_search: PrefixSearch):
    """Positive test: min_score=0.0 returns all videos with any score."""
    results = prefix_search.fuzzy_match_video("эд", 0, None, None, 0.0)
    assert len(results) > 0


def test_min_score_one_returns_empty_or_exact_negative(prefix_search: PrefixSearch):
    """Negative test: min_score=1.0 filters out everything except perfect matches."""
    results = prefix_search.fuzzy_match_video("эдисон", 0, None, None, 1.0)
    for _, score in results:
        assert score >= 1.0


def test_min_score_none_backward_compatible_positive(prefix_search: PrefixSearch):
    """Positive test: min_score=None behaves identically to the original hardcoded 0.01."""
    results_none = prefix_search.fuzzy_match_video("э", 1, None, None, None)
    results_explicit = prefix_search.fuzzy_match_video("э", 1, None, None, 0.001)
    assert results_none == results_explicit



TERMS_CHANNELS = [
    ("эдисон", 123, 10),
    ("эдесон", 124, 5),
    ("эдиссон перец", 5, 8),
    ("эдессон", 126, 7),
    ("эдессо", 127, 5),
]

@pytest.fixture(scope="module")
def channel_search() -> PrefixSearch:
    return prefix_search_builder.from_nodes_video(TERMS_CHANNELS)

def test_channel_exact_match(channel_search: PrefixSearch):
    assert channel_search.fuzzy_match_video("эдисон", 0, 1)[0][1] == 1.0
    assert int(channel_search.fuzzy_match_video("эдисон", 0, 1)[0][0]) == 123
