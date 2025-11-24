use crate::prefix::trie::Trie;

#[derive(Debug, Clone)]
pub struct MatchResult {
    pub node_id: usize,
    pub prefix: String,
    pub distance: u32,
}

#[derive(Debug, Clone)]
pub struct NodeMatch {
    pub node_id: usize,
    pub distance: u16,
}

// Go through trie and collect terminal prefixes with distance <= max_dist
pub fn fuzzy_match(trie: &Trie, term: &str, max_dist: u32, limit: Option<usize>) -> Vec<MatchResult> {
    let term_chars: Vec<char> = term.chars().collect();
    let l = term_chars.len();
    let max_dist_u16 = max_dist as u16;

    // rows[d] — DP-string for depth d (length = l+1)
    let max_depth = l + max_dist as usize + 1;
    let mut rows: Vec<Vec<u16>> = (0..=max_depth).map(|_| vec![0u16; l + 1]).collect();

    for j in 0..=l {
        rows[0][j] = j as u16;
    }

    let mut results: Vec<MatchResult> = Vec::new();
    let mut prefix_buf: Vec<char> = Vec::new();

    fn dfs(
        trie: &Trie,
        node_id: usize,
        depth: usize,
        term_chars: &[char],
        max_dist: u16,
        rows: &mut [Vec<u16>],
        results: &mut Vec<MatchResult>,
        prefix_buf: &mut Vec<char>,
        limit: Option<usize>,
    ) -> bool {
        for cid in trie.children(node_id) {
            let c = trie.child_labels[cid];
            if c == '\0' {
                continue;
            }

            let i = depth + 1;

            let (left, right) = rows.split_at_mut(i);
            let prev = &left[depth];
            let cur = &mut right[0];

            cur[0] = i as u16;

            let mut min_val = cur[0];
            for j in 1..=term_chars.len() {
                let insert_cost = cur[j - 1] + 1;
                let delete_cost = prev[j] + 1;
                let replace_cost = prev[j - 1] + if term_chars[j - 1] == c { 0 } else { 1 };
                let m = insert_cost.min(delete_cost).min(replace_cost);
                cur[j] = m;
                if m < min_val {
                    min_val = m;
                }
            }

            if min_val <= max_dist {
                let last = cur[term_chars.len()];
                // Add only terminal nodes (real keys)
                if trie.terminals[cid] && last <= max_dist {
                    prefix_buf.push(c);
                    let s: String = prefix_buf.iter().collect();
                    results.push(MatchResult {
                        node_id: cid,
                        prefix: s,
                        distance: last as u32,
                    });
                    prefix_buf.pop();

                    if let Some(lim) = limit {
                        if results.len() >= lim {
                            return true; // early stopping
                        }
                    }
                }

                // Go depper
                prefix_buf.push(c);
                if dfs(trie, cid, i, term_chars, max_dist, rows, results, prefix_buf, limit) {
                    prefix_buf.pop();
                    return true;
                }
                prefix_buf.pop();
            }
        }
        false
    }

    let _ = dfs(
        trie,
        0,
        0,
        &term_chars,
        max_dist_u16,
        &mut rows,
        &mut results,
        &mut prefix_buf,
        limit,
    );

    results
}

// Light version for videos: returns only (node_id, distance), without prefix strings.
// only_terminals controls whether to add only terminal nodes.
pub fn fuzzy_match_nodes(
    trie: &Trie,
    term: &str,
    max_dist: u32,
    only_terminals: bool,
    node_limit: Option<usize>,
) -> Vec<NodeMatch> {
    let term_chars: Vec<char> = term.chars().collect();
    let l = term_chars.len();
    let max_dist_u16 = max_dist as u16;

    let max_depth = l + max_dist as usize + 1;
    let mut rows: Vec<Vec<u16>> = (0..=max_depth).map(|_| vec![0u16; l + 1]).collect();
    for j in 0..=l {
        rows[0][j] = j as u16;
    }

    let mut out: Vec<NodeMatch> = Vec::new();

    fn dfs(
        trie: &Trie,
        node_id: usize,
        depth: usize,
        term_chars: &[char],
        max_dist: u16,
        rows: &mut [Vec<u16>],
        out: &mut Vec<NodeMatch>,
        only_terminals: bool,
        node_limit: Option<usize>,
    ) -> bool {
        for cid in trie.children(node_id) {
            let c = trie.child_labels[cid];
            if c == '\0' {
                continue;
            }

            let i = depth + 1;
            let (left, right) = rows.split_at_mut(i);
            let prev = &left[depth];
            let cur = &mut right[0];

            cur[0] = i as u16;

            let mut min_val = cur[0];
            for j in 1..=term_chars.len() {
                let insert_cost = cur[j - 1] + 1;
                let delete_cost = prev[j] + 1;
                let replace_cost = prev[j - 1] + if term_chars[j - 1] == c { 0 } else { 1 };
                let m = insert_cost.min(delete_cost).min(replace_cost);
                cur[j] = m;
                if m < min_val {
                    min_val = m;
                }
            }

            if min_val <= max_dist {
                let last = cur[term_chars.len()];
                let pass = last <= max_dist
                    && (!only_terminals || trie.terminals[cid]);

                if pass {
                    out.push(NodeMatch { node_id: cid, distance: last });
                    if let Some(lim) = node_limit {
                        if out.len() >= lim {
                            return true;
                        }
                    }
                }

                if dfs(
                    trie,
                    cid,
                    i,
                    term_chars,
                    max_dist,
                    rows,
                    out,
                    only_terminals,
                    node_limit,
                ) {
                    return true;
                }
            }
        }
        false
    }

    let _ = dfs(
        trie,
        0,
        0,
        &term_chars,
        max_dist_u16,
        &mut rows,
        &mut out,
        only_terminals,
        node_limit,
    );

    out
}