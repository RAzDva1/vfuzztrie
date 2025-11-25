from typing import Iterable, List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from collections import Counter

from vfuzztrie import PrefixSearch
from fastuuid import UUID as FastUUID

UUIDLike = Union[str, bytes, "FastUUID"]
UUIDOrChannelLike = Union[UUIDLike, int]

@dataclass
class PrefixPack:
    prefix_size: int
    video_ids: List[int]
    video_counts: List[int]

def _uuid_to_16_bytes(u: UUIDLike) -> bytes:
    if FastUUID is not None and isinstance(u, FastUUID):
        return u.bytes
    if isinstance(u, bytes):
        if len(u) != 16:
            raise ValueError("video_uuid bytes must be 16 bytes long")
        return u
    if isinstance(u, str):
        if FastUUID is not None:
            return FastUUID(u).bytes
        hex_str = u.replace("-", "")
        if len(hex_str) != 32:
            raise ValueError(f"Invalid UUID string: {u}")
        return bytes.fromhex(hex_str)
    raise TypeError(f"Unsupported UUID type: {type(u)}")

def _id_to_bytes(x: UUIDOrChannelLike) -> bytes:
    # UUID → 16 bytes, channel_id (int32) → 4 bytes (LE)
    if isinstance(x, int):
        if x < 0 or x > 0xFFFFFFFF:
            raise ValueError(f"channel_id out of uint32 range: {x}")
        return int(x).to_bytes(4, "little", signed=False)
    return _uuid_to_16_bytes(x)

def _build_indices(rows: Iterable[Tuple[str, UUIDOrChannelLike, int]]):
    """
    Creates:
    - video_index: List[bytes] (16 bytes for UUID, 4 bytes for channel_id)
    - query_index: List[str]
    - reverse_indexes
    """
    query_index: List[str] = []
    reverse_query_index: Dict[str, int] = {}

    video_index: List[bytes] = []
    reverse_video_index: Dict[bytes, int] = {}

    # Auto-detect: 'uuid' или 'int32'
    mode: Optional[str] = None

    def get_qid(q: str) -> int:
        qid = reverse_query_index.get(q)
        if qid is None:
            qid = len(query_index)
            query_index.append(q)
            reverse_query_index[q] = qid
        return qid

    def get_vid(u_or_c: UUIDOrChannelLike) -> int:
        nonlocal mode
        cur_is_int = isinstance(u_or_c, int)
        if mode is None:
            mode = "int32" if cur_is_int else "uuid"
        else:
            if (mode == "int32" and not cur_is_int) or (mode == "uuid" and cur_is_int):
                raise ValueError("Mixed id types detected: provide either all UUIDs or all int32 channel_ids in one build.")
        key_bytes = _id_to_bytes(u_or_c)
        vid = reverse_video_index.get(key_bytes)
        if vid is None:
            vid = len(video_index)
            video_index.append(key_bytes)
            reverse_video_index[key_bytes] = vid
        return vid

    # One-pass build without aggregation
    for q, uu_or_ch, _ in rows:
        _ = get_qid(q)
        _ = get_vid(uu_or_ch)

    return query_index, reverse_query_index, video_index, reverse_video_index

def _accumulate_query_prefix(
    rows: Iterable[Tuple[str, UUIDOrChannelLike, int]],
    reverse_query_index: Dict[str, int],
    reverse_video_index: Dict[bytes, int]
) -> Dict[str, PrefixPack]:
    """
    Aggregates statistics by query prefix.
    """
    acc: Dict[str, Tuple[int, Counter[int]]] = {}

    for q, uu_or_ch, cnt in rows:
        vid = reverse_video_index[_id_to_bytes(uu_or_ch)]
        prefix_size, vctr = acc.get(q, (0, Counter()))
        prefix_size += cnt
        vctr[vid] += cnt
        acc[q] = (prefix_size, vctr)

    result: Dict[str, PrefixPack] = {}
    for q, (prefix_size, vctr) in acc.items():
        v_ids, v_counts = zip(*vctr.items()) if vctr else ([], [])
        pack = PrefixPack(
            prefix_size=prefix_size,
            video_ids=list(v_ids),
            video_counts=list(v_counts),
        )
        result[q] = pack
    return result

def _expand_to_trie_nodes(keys_with_payload: List[Tuple[str, Optional[PrefixPack]]]):
    """
    Build arrays for trie nodes
    - node_shifts: List[int]
    - child_labels: List[str] (однобуквенные строки, root — "")
    - payloads: List[Optional[PrefixPack]]
    - child_transitions: List[int]
    """
    indexed_data = [(item[0], item[1], idx) for idx, item in enumerate(keys_with_payload)]

    # add missing prefixes
    prev_key = ""
    sorted_data: List[Tuple[str, Optional[PrefixPack], int]] = []
    for key, payload, orig_index in sorted(indexed_data, key=lambda it: it[0]):
        common_pos = 0
        while (common_pos < min(len(key) - 1, len(prev_key))) and (key[common_pos] == prev_key[common_pos]):
            common_pos += 1

        for key_pos in range(common_pos + 1, len(key)):
            sorted_data.append((key[:key_pos], None, 1_000_000_000))
        sorted_data.append((key, payload, orig_index))
        prev_key = key

    if len(sorted_data) == 0:
        sorted_data = [("", None, 1_000_000_000)]
    elif len(sorted_data[0][0]) > 0:
        sorted_data = [("", None, 1_000_000_000)] + sorted_data

    data_with_ids = [(node_id, key, payload, orig_index)
                     for node_id, (key, payload, orig_index) in enumerate(sorted_data)]

    max_len = max(len(item[0]) for item in sorted_data) if sorted_data else 0
    children_acc: List[List[Tuple[int, str, int]]] = [[] for _ in range(max_len)]
    children_lists: List[List[Tuple[str, int]]] = []

    prev_len = -1
    for node_id, key, payload, orig_index in reversed(data_with_ids):
        if len(key) < prev_len:
            node_children = [(ch[1], ch[0]) for ch in sorted(children_acc[prev_len - 1], key=lambda ch: ch[2])]
            children_lists.append(node_children)
            children_acc[prev_len - 1] = []
        else:
            children_lists.append([])

        if len(key) > 0:
            children_acc[len(key) - 1].append((node_id, key[-1], orig_index))

        prev_len = len(key)

    children_lists = list(reversed(children_lists))

    node_shifts: List[int] = []
    child_transitions: List[int] = []
    child_labels: List[Optional[str]] = [None] * len(data_with_ids)

    for node_children in children_lists:
        node_shifts.append(len(child_transitions))
        for child_label, child_transition in node_children:
            child_labels[child_transition] = child_label
            child_transitions.append(child_transition)

    payloads = [payload for _, _, payload, _ in data_with_ids]

    if child_labels and child_labels[0] is None:
        child_labels[0] = ""

    child_labels = [cl if cl is not None else "" for cl in child_labels]

    return node_shifts, child_labels, payloads, child_transitions

def _pack_to_tuple(pack: PrefixPack | None):
    if pack is None:
        return None
    return (
        int(pack.prefix_size),
        list(map(int, pack.video_ids)),
        list(map(int, pack.video_counts)),
    )

def from_nodes_video(rows: List[Tuple[str, UUIDOrChannelLike, int]]):
    """
    Main index builder with UUID (video) or int32 (channel) support.
    rows: [(query, id, count), ...] where id is UUID-like or int (channel_id).
    Returns PrefixSearch with fuzzy_match_video usable for both cases:
      - UUID mode → returns hyphenated UUID strings
      - Int32 mode → returns decimal channel_id strings
    """
    if not rows:
        node_shifts, child_labels, payloads, child_transitions = [0], [""], [None], []
        return PrefixSearch.from_internal_data_v2(node_shifts, child_labels, payloads, child_transitions, [], [], 10, 25)

    query_index, reverse_query_index, video_index, reverse_video_index = _build_indices(rows)
    by_query_pack = _accumulate_query_prefix(rows, reverse_query_index, reverse_video_index)

    keys_with_payload: List[Tuple[str, Optional[PrefixPack]]] = []
    for q in query_index:
        pack = by_query_pack.get(q)
        keys_with_payload.append((q, pack))

    node_shifts, child_labels, payloads, child_transitions = _expand_to_trie_nodes(keys_with_payload)
    payloads = [_pack_to_tuple(p) for p in payloads]

    return PrefixSearch.from_internal_data_v2(
        node_shifts=node_shifts,
        child_labels=child_labels,
        payloads=payloads,
        child_transitions=child_transitions,
        video_index=video_index,
        video_top_n=25,
    )
