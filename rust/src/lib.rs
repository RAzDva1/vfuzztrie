pub mod prefix;

use crate::prefix::matcher::{fuzzy_match, fuzzy_match_nodes};
use crate::prefix::trie::{Trie, IdMode};

mod payload;
use payload::Payload;

use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn payload_from_py(tup: &pyo3::Bound<'_, pyo3::types::PyTuple>) -> PyResult<Payload> {
    if tup.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("Payload tuple must have 3 elements"));
    }
    let prefix_size: u32 = tup.get_item(0)?.extract()?;
    let video_ids: Vec<u32> = tup.get_item(1)?.extract()?;
    let video_counts: Vec<u32> = tup.get_item(2)?.extract()?;

    if video_ids.len() != video_counts.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "video_ids and video_counts length mismatch",
        ));
    }

    Ok(Payload {
        prefix_size,
        video_ids,
        video_counts,
    })
}

#[pyclass]
#[derive(Clone)]
pub struct PrefixSearch {
    trie: Trie,
}

#[pymethods]
impl PrefixSearch {
    #[staticmethod]
    #[pyo3(signature = (
        node_shifts,
        child_labels,
        payloads,
        child_transitions,
        video_index,
        video_top_n
    ))]
    pub fn from_internal_data_v2(
        py: Python,
        node_shifts: Vec<u32>,
        child_labels: Vec<String>,
        payloads: Vec<Option<PyObject>>,
        child_transitions: Vec<u32>,
        video_index: Vec<Vec<u8>>,
        video_top_n: usize,
    ) -> PyResult<Self> {
        let n = child_labels.len();
        if node_shifts.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "node_shifts len {} != child_labels len {}",
                node_shifts.len(),
                n
            )));
        }
        if payloads.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "payloads len {} != child_labels len {}",
                payloads.len(),
                n
            )));
        }
        for (i, &cid) in child_transitions.iter().enumerate() {
            if (cid as usize) >= n {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "child_transitions[{}] = {} out of range [0..{})",
                    i, cid, n
                )));
            }
        }

        // labels → char
        let mut child_labels_char: Vec<char> = Vec::with_capacity(child_labels.len());
        for s in child_labels {
            child_labels_char.push(s.chars().next().unwrap_or('\0'));
        }

        // terminal flags before payload transformation
        let terminal_flags: Vec<bool> = payloads.iter().map(|opt| opt.is_some()).collect();

        // payloads: expects Tuple( prefix_size, v_ids, v_counts )
        let mut rust_payloads: Vec<Option<Payload>> = Vec::with_capacity(payloads.len());
        for opt in payloads {
            if let Some(obj) = opt {
                let any = obj.bind(py);
                let tup = any.downcast::<pyo3::types::PyTuple>()?;
                let p = payload_from_py(&tup)?;
                rust_payloads.push(Some(p));
            } else {
                rust_payloads.push(None);
            }
        }

        // Autodetect id mode by element size: 16 → UUID, 4 → int32
        let mut id_mode = IdMode::Uuid16;
        let mut video_index_fixed: Vec<[u8; 16]> = Vec::new();
        let mut channel_index_fixed: Vec<u32> = Vec::new();

        if !video_index.is_empty() {
            let all_16 = video_index.iter().all(|v| v.len() == 16);
            let all_4 = video_index.iter().all(|v| v.len() == 4);

            if all_16 {
                id_mode = IdMode::Uuid16;
                video_index_fixed = Vec::with_capacity(video_index.len());
                for v in video_index {
                    let mut arr = [0u8; 16];
                    arr.copy_from_slice(&v);
                    video_index_fixed.push(arr);
                }
            } else if all_4 {
                id_mode = IdMode::Int32;
                channel_index_fixed = Vec::with_capacity(video_index.len());
                for v in video_index {
                    let a: [u8; 4] = v
                        .as_slice()
                        .try_into()
                        .map_err(|_| pyo3::exceptions::PyValueError::new_err("int32 id must be 4 bytes"))?;
                    channel_index_fixed.push(u32::from_le_bytes(a));
                }
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "video_index entries must be all 16 bytes (UUID) or all 4 bytes (int32)",
                ));
            }
        }

        let mut trie = Trie {
            node_shifts,
            child_labels: child_labels_char,
            child_transitions,
            payloads: rust_payloads,
            video_index: video_index_fixed,
            channel_index: channel_index_fixed,
            id_mode,
            terminals: terminal_flags,
        };

        trie.propagate(video_top_n);
        Ok(Self { trie })
    }

    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyResult<pyo3::Bound<'py, PyBytes>> {
        let cfg = bincode::config::standard();
        let data = bincode::encode_to_vec(&self.trie, cfg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("encode error: {e}")))?;
        Ok(PyBytes::new(py, &data))
    }

    #[staticmethod]
    pub fn from_bytes(b: pyo3::Bound<'_, PyBytes>) -> PyResult<Self> {
        let slice = b.as_bytes();
        let cfg = bincode::config::standard();
        let (trie, _): (Trie, usize) = bincode::decode_from_slice(slice, cfg)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("decode error: {e}")))?;
        Ok(Self { trie })
    }

    #[pyo3(signature = (term, max_dist = 1, limit = None))]
    #[pyo3(text_signature = "(term, max_dist=1, limit=None)")]
    pub fn fuzzy_match(&self, term: &str, max_dist: u32, limit: Option<usize>) -> Vec<(String, i32, u32)> {
        let matches = fuzzy_match(&self.trie, term, max_dist, limit);
        matches
            .into_iter()
            .map(|m| {
                let has_payload = self.trie.terminals[m.node_id] as i32;
                (m.prefix, has_payload, m.distance)
            })
            .collect()
    }

    #[pyo3(signature = (term, max_dist = 1, limit = None, include_prefix_nodes = None))]
    #[pyo3(text_signature = "(term, max_dist=1, limit=None, include_prefix_nodes=None)")]
    pub fn fuzzy_match_video(
        &self,
        term: &str,
        max_dist: u32,
        limit: Option<usize>,
        include_prefix_nodes: Option<bool>,
    ) -> Vec<(String, f64)> {
        use std::cmp::Ordering;
        use std::collections::HashMap;

        let node_matches = match include_prefix_nodes {
            Some(true) => fuzzy_match_nodes(&self.trie, term, max_dist, false, None),
            Some(false) => fuzzy_match_nodes(&self.trie, term, max_dist, true, None),
            None => {
                let m = fuzzy_match_nodes(&self.trie, term, max_dist, true, None);
                if m.is_empty() {
                    fuzzy_match_nodes(&self.trie, term, max_dist, false, None)
                } else {
                    m
                }
            }
        };

        let term_len = term.chars().count() as f64;
        let mut total_prefix_size: f64 = 0.0;
        let mut video_scores: HashMap<u32, f64> =
            HashMap::with_capacity(node_matches.len().saturating_mul(16));

        const NONTERM_PENALTY: f64 = 0.85;

        for m in node_matches.iter() {
            let mut coeff = if m.distance == 0 {
                1.0
            } else if term_len == 0.0 {
                0.0
            } else {
                0.05 * (1.0 - (m.distance as f64) / term_len)
            };
            if coeff <= 0.0 {
                continue;
            }

            if !self.trie.terminals[m.node_id] {
                coeff *= NONTERM_PENALTY;
            }

            if let Some(ref payload) = self.trie.payloads[m.node_id] {
                total_prefix_size += coeff * (payload.prefix_size as f64);
                let ids = &payload.video_ids;
                let cnts = &payload.video_counts;
                for (i, vid) in ids.iter().enumerate() {
                    let cnt = cnts[i] as f64;
                    *video_scores.entry(*vid).or_insert(0.0) += coeff * cnt;
                }
            }
        }

        if total_prefix_size <= 0.0 {
            return Vec::new();
        }

        let mut scored: Vec<(u32, f64)> = video_scores
            .into_iter()
            .map(|(vid, sc)| (vid, sc / total_prefix_size))
            .filter(|&(_, norm)| norm >= 0.01)
            .collect();

        if let Some(k) = limit {
            if scored.len() > k {
                let _ = scored.select_nth_unstable_by(k, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
                });
                scored.truncate(k);
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            } else {
                scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
            }
        } else {
            scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        }

        let mut out: Vec<(String, f64)> = Vec::with_capacity(scored.len());
        match self.trie.id_mode {
            IdMode::Uuid16 => {
                for (vid, norm) in scored {
                    if let Some(uuid_bytes) = self.trie.video_index.get(vid as usize) {
                        let s = uuid_to_hyphenated_string(uuid_bytes);
                        out.push((s, norm));
                    }
                }
            }
            IdMode::Int32 => {
                for (vid, norm) in scored {
                    if let Some(ch_id) = self.trie.channel_index.get(vid as usize) {
                        out.push((format!("{}", ch_id), norm));
                    }
                }
            }
        }
        out
    }
}

fn uuid_to_hyphenated_string(bytes16: &[u8; 16]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(36);
    let parts = [
        &bytes16[0..4],
        &bytes16[4..6],
        &bytes16[6..8],
        &bytes16[8..10],
        &bytes16[10..16],
    ];
    for (i, part) in parts.iter().enumerate() {
        for byte in *part {
            write!(&mut s, "{:02x}", byte).unwrap();
        }
        if i != parts.len() - 1 {
            s.push('-');
        }
    }
    s
}

#[pymodule]
fn _vfuzztrie(py: Python, m: &pyo3::Bound<pyo3::types::PyModule>) -> PyResult<()> {
    m.add("PrefixSearch", py.get_type::<PrefixSearch>())?;
    Ok(())
}