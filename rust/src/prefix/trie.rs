use bincode::{Decode, Encode};

use crate::payload::Payload;

#[derive(Debug, Clone, Encode, Decode)]
pub enum IdMode {
    Uuid16,
    Int32,
}

#[derive(Debug, Clone, Encode, Decode)]
pub struct Trie {
    pub node_shifts: Vec<u32>,
    pub child_labels: Vec<char>,
    pub child_transitions: Vec<u32>,
    pub payloads: Vec<Option<Payload>>,
    // If id_mode == Uuid16 → use video_index
    pub video_index: Vec<[u8; 16]>,
    // If id_mode == Int32 → use channel_index
    pub channel_index: Vec<u32>,
    pub id_mode: IdMode,
    pub terminals: Vec<bool>,
}

impl Trie {
    pub fn children_range(&self, node_id: usize) -> std::ops::Range<usize> {
        let start = self.node_shifts[node_id] as usize;
        let end = if node_id + 1 < self.node_shifts.len() {
            self.node_shifts[node_id + 1] as usize
        } else {
            self.child_transitions.len()
        };
        start..end
    }

    pub fn children(&self, node_id: usize) -> impl Iterator<Item = usize> + '_ {
        self.children_range(node_id)
            .map(|i| self.child_transitions[i] as usize)
    }

    pub fn propagate(&mut self, video_top_n: usize) {
        let n = self.child_labels.len();

        for p in self.payloads.iter_mut() {
            if p.is_none() {
                *p = Some(Payload::empty());
            }
        }

        for node_id in (0..n).rev() {
            let mut agg = self.payloads[node_id].take().unwrap();
            for ch in self.children(node_id) {
                if let Some(ref child_p) = self.payloads[ch] {
                    let tmp = child_p.clone();
                    agg.merge_from(&tmp);
                }
            }
            agg.prune(Some(video_top_n));
            self.payloads[node_id] = Some(agg);
        }
    }
}