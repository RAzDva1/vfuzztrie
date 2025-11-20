use crate::prefix::trie::Trie;

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub node_id: usize,
    pub prefix: String,
    pub distance: u32,
}

pub fn fuzzy_match(trie: &Trie, term: &str, max_dist: u32, limit: Option<usize>) -> Vec<MatchResult> {
    #[derive(Clone)]
    struct State {
        visited: bool,
        cur_prefix: String,
        prev_row: Vec<u32>,
        cur_row: Vec<u32>,
    }

    let mut results: Vec<MatchResult> = Vec::new();
    let mut stack: Vec<(usize, State)> = Vec::new();
    let root_state = State {
        visited: false,
        cur_prefix: String::new(),
        prev_row: vec![],
        cur_row: vec![],
    };
    stack.push((0, root_state));

    while let Some((node_id, mut st)) = stack.pop() {
        if !st.visited {
            if st.cur_prefix.is_empty() {
                st.cur_row = (0..=term.len() as u32).collect();
            } else {
                let mut cur_row: Vec<u32> = Vec::with_capacity(term.len() + 1);
                cur_row.push(st.prev_row[0] + 1);
                let last_char = st.cur_prefix.chars().last().unwrap_or('\0');

                for (col, tc) in term.chars().enumerate() {
                    let insert_cost = cur_row[col] + 1;
                    let delete_cost = st.prev_row[col + 1] + 1;
                    let replace_cost = if tc != last_char { st.prev_row[col] + 1 } else { st.prev_row[col] };
                    let m = insert_cost.min(delete_cost).min(replace_cost);
                    cur_row.push(m);
                }
                st.cur_row = cur_row;

                if trie.payloads[node_id].is_some() {
                    let last = *st.cur_row.last().unwrap_or(&u32::MAX);
                    if last <= max_dist {
                        results.push(MatchResult {
                            node_id,
                            prefix: st.cur_prefix.clone(),
                            distance: last,
                        });
                    }
                }
            }

            st.visited = true;
            stack.push((node_id, st.clone()));

            let min_in_row = st.cur_row.iter().min().cloned().unwrap_or(u32::MAX);
            if min_in_row <= max_dist {
                let mut chs: Vec<(char, usize)> = trie
                    .children(node_id)
                    .map(|id| (trie.child_labels[id], id))
                    .collect();
                chs.sort_unstable_by(|a, b| a.0.cmp(&b.0));

                for (_, cid) in chs.into_iter().rev() {
                    let child_state = State {
                        visited: false,
                        cur_prefix: {
                            let mut s = st.cur_prefix.clone();
                            if trie.child_labels[cid] != '\0' {
                                s.push(trie.child_labels[cid]);
                            }
                            s
                        },
                        prev_row: st.cur_row.clone(),
                        cur_row: Vec::new(),
                    };
                    stack.push((cid, child_state));
                }
            }
        }
    }

    if let Some(lim) = limit {
        results.truncate(lim);
    }
    results
}