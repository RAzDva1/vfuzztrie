from vfuzztrie import prefix_search_builder, PrefixSearch
import pytest

TERMS = [
    ("эдисон", "123e4567-e89b-12d3-a456-426614174000", 10),
    ("эдесон", "123e4567-e89b-12d3-a456-426614174001", 5),
    ("эдиссон перец", "123e4567-e89b-12d3-a456-426614174002", 8),
    ("эдессон", "123e4567-e89b-12d3-a456-426614174003", 7),
    ("эдессо", "123e4567-e89b-12d3-a456-426614174004", 6),
]


@pytest.fixture(scope="module")
def prefix_search() -> PrefixSearch:
    return prefix_search_builder.from_nodes_video(TERMS)


def test_eq_search(prefix_search: PrefixSearch):
    prefix_search_de: PrefixSearch = PrefixSearch.from_bytes(prefix_search.to_bytes())

    assert prefix_search_de.fuzzy_match_video("эдессон", 1, None) == prefix_search.fuzzy_match_video("эдессон", 1, None)
