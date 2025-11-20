use bincode::{Decode, Encode};
use std::collections::HashMap;

#[derive(Debug, Clone, Encode, Decode)]
pub struct Payload {
    pub prefix_size: u32,
    pub video_ids: Vec<u32>,
    pub video_counts: Vec<u32>,
}

impl Payload {
    pub fn empty() -> Self {
        Self {
            prefix_size: 0,
            video_ids: Vec::new(),
            video_counts: Vec::new(),
        }
    }

    pub fn merge_from(&mut self, other: &Payload) {
        self.prefix_size = self.prefix_size.saturating_add(other.prefix_size);

        let mut vmap: HashMap<u32, u64> = HashMap::new();

        for (i, id) in self.video_ids.iter().enumerate() {
            *vmap.entry(*id).or_insert(0) += self.video_counts[i] as u64;
        }
        for (i, id) in other.video_ids.iter().enumerate() {
            *vmap.entry(*id).or_insert(0) += other.video_counts[i] as u64;
        }

        let mut vpairs: Vec<(u32, u64)> = vmap.into_iter().collect();
        vpairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        self.video_ids = vpairs.iter().map(|(id, _)| *id).collect();
        self.video_counts = vpairs.iter().map(|(_, c)| *c as u32).collect();
    }


    pub fn prune(&mut self, video_top_n: Option<usize>) {
        if let Some(n) = video_top_n {
            if self.video_ids.len() > n {
                self.video_ids.truncate(n);
                self.video_counts.truncate(n);
            }
        }
    }
}