use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

type Token = i32;
type NodeId = usize;
type ChildKey = Vec<Token>;

const ROOT: NodeId = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeInfo {
    pub evictable_size: usize,
    pub protected_size: usize,
}

impl SizeInfo {
    pub fn total_size(self) -> usize {
        self.evictable_size + self.protected_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadixCacheHandle {
    pub cached_len: usize,
    node: NodeId,
}

impl RadixCacheHandle {
    pub fn node_id(self) -> usize {
        self.node
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchResult {
    pub handle: RadixCacheHandle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertResult {
    /// The length that was already present before this insertion.
    pub cached_len: usize,
    /// A handle pointing to the inserted / matched full aligned prefix.
    pub handle: RadixCacheHandle,
}

#[derive(Debug, Clone)]
struct RadixTreeNode {
    key: Vec<Token>,
    value: Vec<Token>,
    children: HashMap<ChildKey, NodeId>,
    parent: Option<NodeId>,
    ref_count: usize,
    uuid: usize,
    timestamp: u64,
    alive: bool,
}

impl RadixTreeNode {
    fn new_root(timestamp: u64) -> Self {
        Self {
            key: Vec::new(),
            value: Vec::new(),
            children: HashMap::new(),
            parent: None,
            ref_count: 1,
            uuid: ROOT,
            timestamp,
            alive: true,
        }
    }

    fn new(
        key: Vec<Token>,
        value: Vec<Token>,
        parent: NodeId,
        uuid: usize,
        timestamp: u64,
    ) -> Self {
        assert_eq!(key.len(), value.len(), "key/value lengths must match");
        assert!(
            !key.is_empty(),
            "non-root radix nodes must have non-empty keys"
        );
        Self {
            key,
            value,
            children: HashMap::new(),
            parent: Some(parent),
            ref_count: 0,
            uuid,
            timestamp,
            alive: true,
        }
    }

    fn len(&self) -> usize {
        self.key.len()
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct EvictCandidate {
    timestamp: u64,
    uuid: usize,
    node: NodeId,
}

impl Ord for EvictCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is max-first; reverse ordering to pop the oldest timestamp first.
        other
            .timestamp
            .cmp(&self.timestamp)
            .then_with(|| other.uuid.cmp(&self.uuid))
    }
}

impl PartialOrd for EvictCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A Rust port of `minisgl.kvcache.radix_cache.RadixPrefixCache`.
///
/// The implementation mirrors the Python version:
/// - compressed radix-tree edges store `key` token spans and `value` cache indices;
/// - matching/insertion are aligned down to `page_size`;
/// - handles can be locked/unlocked through ancestor ref-counts;
/// - eviction picks unlocked leaf nodes by LRU timestamp and may evict more than requested.
#[derive(Debug, Clone)]
pub struct RadixPrefixCache {
    page_size: usize,
    nodes: Vec<RadixTreeNode>,
    evictable_size: usize,
    protected_size: usize,
    next_timestamp: u64,
}

impl RadixPrefixCache {
    pub fn new(page_size: usize) -> Self {
        assert!(page_size > 0, "page_size must be positive");
        Self {
            page_size,
            nodes: vec![RadixTreeNode::new_root(0)],
            evictable_size: 0,
            protected_size: 0,
            next_timestamp: 1,
        }
    }

    pub fn page_size(&self) -> usize {
        self.page_size
    }

    pub fn size_info(&self) -> SizeInfo {
        SizeInfo {
            evictable_size: self.evictable_size,
            protected_size: self.protected_size,
        }
    }

    pub fn match_prefix(&mut self, input_ids: &[Token]) -> MatchResult {
        let (node, prefix_len) = self.tree_walk(input_ids);
        MatchResult {
            handle: RadixCacheHandle {
                cached_len: prefix_len,
                node,
            },
        }
    }

    pub fn insert_prefix(&mut self, input_ids: &[Token], indices: &[Token]) -> InsertResult {
        assert_eq!(
            input_ids.len(),
            indices.len(),
            "input_ids/indices lengths must match"
        );
        let insert_len = align_down(input_ids.len(), self.page_size);
        let input_ids = &input_ids[..insert_len];
        let indices = &indices[..insert_len];

        let (mut node, prefix_len) = self.tree_walk(input_ids);
        if prefix_len != insert_len {
            let new_node = self.add_child(
                node,
                input_ids[prefix_len..].to_vec(),
                indices[prefix_len..].to_vec(),
            );
            self.evictable_size += self.nodes[new_node].len();
            node = new_node;
        }

        InsertResult {
            cached_len: prefix_len,
            handle: RadixCacheHandle {
                cached_len: insert_len,
                node,
            },
        }
    }

    pub fn lock_handle(&mut self, handle: RadixCacheHandle) {
        let mut node = handle.node;
        while node != ROOT {
            if self.nodes[node].ref_count == 0 {
                self.evictable_size -= self.nodes[node].len();
                self.protected_size += self.nodes[node].len();
            }
            self.nodes[node].ref_count += 1;
            node = self.nodes[node]
                .parent
                .expect("non-root node must have parent");
        }
    }

    pub fn unlock_handle(&mut self, handle: RadixCacheHandle) {
        let mut node = handle.node;
        while node != ROOT {
            assert!(
                self.nodes[node].ref_count > 0,
                "unlock would make ref_count negative"
            );
            self.nodes[node].ref_count -= 1;
            if self.nodes[node].ref_count == 0 {
                self.evictable_size += self.nodes[node].len();
                self.protected_size -= self.nodes[node].len();
            }
            node = self.nodes[node]
                .parent
                .expect("non-root node must have parent");
        }
    }

    pub fn get_matched_indices(&self, handle: RadixCacheHandle) -> Vec<Token> {
        let mut node = handle.node;
        let mut parts: Vec<&[Token]> = Vec::new();
        while node != ROOT {
            let n = &self.nodes[node];
            assert!(n.alive, "handle points to an evicted node");
            parts.push(&n.value);
            node = n.parent.expect("non-root node must have parent");
        }
        parts.reverse();

        let mut result = Vec::with_capacity(handle.cached_len);
        for part in parts {
            result.extend_from_slice(part);
        }
        result.truncate(handle.cached_len);
        result
    }

    pub fn evict(&mut self, size: usize) -> Vec<Token> {
        if size == 0 {
            return Vec::new();
        }
        assert!(
            size <= self.evictable_size,
            "Cannot evict {}, only {} is evictable",
            size,
            self.evictable_size
        );

        let mut heap = BinaryHeap::new();
        self.collect_leaf_nodes_for_evict(&mut heap);

        let mut evicted_indices = Vec::new();
        let mut evicted_size = 0;

        while evicted_size < size {
            let candidate = heap
                .pop()
                .expect("Cannot evict enough cache: candidate heap exhausted");
            let node = candidate.node;

            if !self.is_evictable_leaf(node) {
                continue;
            }

            evicted_size += self.nodes[node].len();
            evicted_indices.extend_from_slice(&self.nodes[node].value);
            self.evictable_size -= self.nodes[node].len();

            let parent = self.nodes[node]
                .parent
                .expect("evicted node must not be root");
            let child_key = self.child_key(&self.nodes[node].key);
            self.nodes[parent].children.remove(&child_key);
            self.nodes[node].alive = false;

            if self.is_evictable_leaf(parent) {
                heap.push(EvictCandidate {
                    timestamp: self.nodes[parent].timestamp,
                    uuid: self.nodes[parent].uuid,
                    node: parent,
                });
            }
        }

        evicted_indices
    }

    pub fn reset(&mut self) {
        *self = Self::new(self.page_size);
    }

    pub fn check_integrity(&self) -> Result<(), String> {
        if self.nodes[ROOT].parent.is_some() || self.nodes[ROOT].ref_count != 1 {
            return Err("root must have no parent and ref_count=1".to_string());
        }

        let mut visited = HashSet::new();
        let mut evictable = 0;
        let mut protected = 0;
        self.check_node(ROOT, &mut visited, &mut evictable, &mut protected)?;

        if evictable != self.evictable_size || protected != self.protected_size {
            return Err(format!(
                "size mismatch: computed evictable/protected=({evictable},{protected}), stored=({},{})",
                self.evictable_size, self.protected_size
            ));
        }
        Ok(())
    }

    pub fn debug_dump(&self) -> Vec<(usize, Option<usize>, Vec<Token>, Vec<Token>, usize)> {
        self.nodes
            .iter()
            .filter(|n| n.alive)
            .map(|n| {
                (
                    n.uuid,
                    n.parent,
                    n.key.clone(),
                    n.value.clone(),
                    n.ref_count,
                )
            })
            .collect()
    }

    fn add_child(&mut self, parent: NodeId, key: Vec<Token>, value: Vec<Token>) -> NodeId {
        let child_key = self.child_key(&key);
        let node = self.nodes.len();
        let timestamp = self.tick();
        self.nodes
            .push(RadixTreeNode::new(key, value, parent, node, timestamp));
        let replaced = self.nodes[parent].children.insert(child_key, node);
        assert!(
            replaced.is_none(),
            "child key collision while inserting radix node"
        );
        node
    }

    fn tree_walk(&mut self, input_ids: &[Token]) -> (NodeId, usize) {
        let mut prefix_len = 0;
        let total_len = input_ids.len();
        let mut node = ROOT;
        let tic = self.tick();

        while prefix_len < total_len {
            let key = self.child_key(&input_ids[prefix_len..]);
            let Some(child) = self.nodes[node].children.get(&key).copied() else {
                return (node, prefix_len);
            };
            node = child;

            let match_len = align_down(
                common_prefix_len(&self.nodes[node].key, &input_ids[prefix_len..]),
                self.page_size,
            );
            prefix_len += match_len;

            if match_len != self.nodes[node].len() {
                let split_node = self.split_at(node, match_len, tic);
                return (split_node, prefix_len);
            }

            self.nodes[node].timestamp = tic;
        }

        (node, prefix_len)
    }

    fn split_at(&mut self, node: NodeId, pos: usize, timestamp: u64) -> NodeId {
        assert!(
            pos > 0 && pos < self.nodes[node].len(),
            "invalid split position"
        );
        let parent = self.nodes[node].parent.expect("cannot split root");

        let old_key = self.nodes[node].key.clone();
        let old_value = self.nodes[node].value.clone();
        let old_ref_count = self.nodes[node].ref_count;
        let old_child_key = self.child_key(&old_key);

        let new_parent_id = self.nodes.len();
        let mut new_parent = RadixTreeNode::new(
            old_key[..pos].to_vec(),
            old_value[..pos].to_vec(),
            parent,
            new_parent_id,
            timestamp,
        );
        new_parent.ref_count = old_ref_count;

        self.nodes.push(new_parent);
        self.nodes[parent]
            .children
            .insert(old_child_key, new_parent_id);

        self.nodes[node].key = old_key[pos..].to_vec();
        self.nodes[node].value = old_value[pos..].to_vec();
        self.nodes[node].parent = Some(new_parent_id);

        let new_child_key = self.child_key(&self.nodes[node].key);
        self.nodes[new_parent_id]
            .children
            .insert(new_child_key, node);

        new_parent_id
    }

    fn child_key(&self, tokens: &[Token]) -> ChildKey {
        assert!(!tokens.is_empty(), "child key requires at least one token");
        tokens[..tokens.len().min(self.page_size)].to_vec()
    }

    fn tick(&mut self) -> u64 {
        let value = self.next_timestamp;
        self.next_timestamp += 1;
        value
    }

    fn is_evictable_leaf(&self, node: NodeId) -> bool {
        node != ROOT
            && self.nodes[node].alive
            && self.nodes[node].ref_count == 0
            && self.nodes[node].is_leaf()
    }

    fn collect_leaf_nodes_for_evict(&self, heap: &mut BinaryHeap<EvictCandidate>) {
        let mut stack = vec![ROOT];
        while let Some(node) = stack.pop() {
            if self.is_evictable_leaf(node) {
                heap.push(EvictCandidate {
                    timestamp: self.nodes[node].timestamp,
                    uuid: self.nodes[node].uuid,
                    node,
                });
            } else {
                for child in self.nodes[node].children.values() {
                    stack.push(*child);
                }
            }
        }
    }

    fn check_node(
        &self,
        node: NodeId,
        visited: &mut HashSet<NodeId>,
        evictable: &mut usize,
        protected: &mut usize,
    ) -> Result<(), String> {
        if !visited.insert(node) {
            return Err(format!("cycle or duplicate child reference at node {node}"));
        }
        if !self.nodes[node].alive {
            return Err(format!("reachable node {node} is marked dead"));
        }

        if node != ROOT {
            let n = &self.nodes[node];
            if n.key.is_empty() || n.key.len() != n.value.len() {
                return Err(format!("invalid key/value length at node {node}"));
            }
            if n.key.len() % self.page_size != 0 {
                return Err(format!("node {node} length is not page aligned"));
            }
            if n.ref_count == 0 {
                *evictable += n.len();
            } else {
                *protected += n.len();
            }
        }

        for (key, child) in &self.nodes[node].children {
            if self.nodes[*child].parent != Some(node) {
                return Err(format!("child {child} parent mismatch"));
            }
            let expected_key = self.child_key(&self.nodes[*child].key);
            if key != &expected_key {
                return Err(format!("child map key mismatch for child {child}"));
            }
            self.check_node(*child, visited, evictable, protected)?;
        }
        Ok(())
    }
}

fn align_down(value: usize, alignment: usize) -> usize {
    value / alignment * alignment
}

fn common_prefix_len(lhs: &[Token], rhs: &[Token]) -> usize {
    lhs.iter()
        .zip(rhs.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_match_full_prefix_page_size_one() {
        let mut cache = RadixPrefixCache::new(1);
        let inserted = cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]);

        assert_eq!(inserted.cached_len, 0);
        assert_eq!(inserted.handle.cached_len, 3);
        assert_eq!(cache.size_info().evictable_size, 3);

        let matched = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(matched.handle.cached_len, 3);
        assert_eq!(cache.get_matched_indices(matched.handle), vec![10, 11, 12]);
        cache.check_integrity().unwrap();
    }

    #[test]
    fn insert_splits_partial_node_on_page_boundary() {
        let mut cache = RadixPrefixCache::new(1);
        cache.insert_prefix(&[1, 2, 3, 4], &[10, 11, 12, 13]);

        let matched = cache.match_prefix(&[1, 2, 9]);
        assert_eq!(matched.handle.cached_len, 2);
        assert_eq!(cache.get_matched_indices(matched.handle), vec![10, 11]);

        let inserted = cache.insert_prefix(&[1, 2, 9], &[10, 11, 99]);
        assert_eq!(inserted.cached_len, 2);

        let old_branch = cache.match_prefix(&[1, 2, 3, 4]);
        assert_eq!(old_branch.handle.cached_len, 4);
        assert_eq!(
            cache.get_matched_indices(old_branch.handle),
            vec![10, 11, 12, 13]
        );

        let new_branch = cache.match_prefix(&[1, 2, 9]);
        assert_eq!(new_branch.handle.cached_len, 3);
        assert_eq!(
            cache.get_matched_indices(new_branch.handle),
            vec![10, 11, 99]
        );
        cache.check_integrity().unwrap();
    }

    #[test]
    fn page_size_alignment_drops_tail() {
        let mut cache = RadixPrefixCache::new(4);
        let inserted = cache.insert_prefix(&[0, 1, 2, 3, 4, 5], &[100, 101, 102, 103, 104, 105]);

        assert_eq!(inserted.handle.cached_len, 4);
        assert_eq!(cache.size_info().evictable_size, 4);

        let matched = cache.match_prefix(&[0, 1, 2, 3, 99, 100]);
        assert_eq!(matched.handle.cached_len, 4);
        assert_eq!(
            cache.get_matched_indices(matched.handle),
            vec![100, 101, 102, 103]
        );
        cache.check_integrity().unwrap();
    }

    #[test]
    fn lock_and_unlock_updates_size_info_along_path() {
        let mut cache = RadixPrefixCache::new(1);
        let handle = cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]).handle;

        cache.lock_handle(handle);
        assert_eq!(
            cache.size_info(),
            SizeInfo {
                evictable_size: 0,
                protected_size: 3
            }
        );

        cache.unlock_handle(handle);
        assert_eq!(
            cache.size_info(),
            SizeInfo {
                evictable_size: 3,
                protected_size: 0
            }
        );
        cache.check_integrity().unwrap();
    }

    #[test]
    fn evict_oldest_unprotected_leaf() {
        let mut cache = RadixPrefixCache::new(1);
        cache.insert_prefix(&[1, 2], &[10, 11]);
        cache.insert_prefix(&[3, 4], &[30, 31]);

        let evicted = cache.evict(1);
        assert_eq!(evicted, vec![10, 11]);
        assert_eq!(cache.size_info().evictable_size, 2);

        assert_eq!(cache.match_prefix(&[1, 2]).handle.cached_len, 0);
        assert_eq!(cache.match_prefix(&[3, 4]).handle.cached_len, 2);
        cache.check_integrity().unwrap();
    }

    #[test]
    fn evict_promotes_unlocked_parent_after_leaf_removed() {
        let mut cache = RadixPrefixCache::new(1);
        cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]);
        cache.insert_prefix(&[1, 2, 4], &[10, 11, 14]);

        assert_eq!(cache.match_prefix(&[1, 2]).handle.cached_len, 2);
        let evicted = cache.evict(3);

        assert_eq!(evicted.len(), 4);
        assert_eq!(cache.size_info().evictable_size, 0);
        assert_eq!(cache.match_prefix(&[1, 2, 3]).handle.cached_len, 0);
        cache.check_integrity().unwrap();
    }

    #[test]
    fn locked_handle_cannot_be_evicted() {
        let mut cache = RadixPrefixCache::new(1);
        let locked = cache.insert_prefix(&[1, 2], &[10, 11]).handle;
        cache.insert_prefix(&[3, 4], &[30, 31]);
        cache.lock_handle(locked);

        let evicted = cache.evict(2);
        assert_eq!(evicted, vec![30, 31]);

        assert_eq!(cache.match_prefix(&[1, 2]).handle.cached_len, 2);
        cache.unlock_handle(locked);
        cache.check_integrity().unwrap();
    }

    #[test]
    fn reset_clears_cache() {
        let mut cache = RadixPrefixCache::new(2);
        cache.insert_prefix(&[1, 2, 3, 4], &[10, 11, 12, 13]);
        cache.reset();

        assert_eq!(cache.size_info().total_size(), 0);
        assert_eq!(cache.match_prefix(&[1, 2]).handle.cached_len, 0);
        cache.check_integrity().unwrap();
    }
}
