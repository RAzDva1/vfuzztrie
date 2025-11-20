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
    assert set(res[0] for res in prefix_search.fuzzy_match_video("эдессон", 1, None)) == {'123e4567-e89b-12d3-a456-426614174001', '123e4567-e89b-12d3-a456-426614174002', '123e4567-e89b-12d3-a456-426614174003', '123e4567-e89b-12d3-a456-426614174004' }


def test_fuzzy_match_distance_eq_1_with_limit(prefix_search: PrefixSearch):
    assert prefix_search.fuzzy_match_video("идисон", 1, 1)[0][0] == "123e4567-e89b-12d3-a456-426614174000"


def test_fuzzy_match_distance_eq_2(prefix_search: PrefixSearch):
    assert set(res[0] for res in prefix_search.fuzzy_match_video("эдисссонн", 2, None)) == {"123e4567-e89b-12d3-a456-426614174002"}

