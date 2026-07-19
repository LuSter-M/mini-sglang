use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};

/// Token id / KV cache index 的基础整数类型。
///
/// 为了贴近 mini-sglang 里 `torch.int32` token/index tensor 的语义，这里统一用 `i32`。
pub type Token = i32;
type NodeId = usize;
type ChildKey = Vec<Token>;

const ROOT: NodeId = 0;

/// 当前 prefix cache 的容量统计。
///
/// 这里的 size 单位是 token/index 个数，而不是节点数或 page 数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SizeInfo {
    /// 可以被 evict 的 token 数量：所有 `ref_count == 0` 的可达非 root 节点长度之和。
    pub evictable_size: usize,
    /// 正在被 handle lock 保护、不能 evict 的 token 数量。
    pub protected_size: usize,
}

impl SizeInfo {
    pub fn total_size(self) -> usize {
        self.evictable_size + self.protected_size
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RadixCacheHandle {
    /// 从 root 到 `node` 这条路径上已经命中的 prefix 长度。
    pub cached_len: usize,
    /// handle 指向的 radix tree 节点。节点 id 不对外暴露可变访问。
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
    /// 该压缩边保存的 token 片段。
    key: Vec<Token>,
    /// 与 `key` 一一对应的 KV cache index 片段。
    value: Vec<Token>,
    /// 子节点索引。key 是子节点 `key` 的第一个 page，因此查找可以按 page 前缀跳转。
    children: HashMap<ChildKey, NodeId>,
    /// 父节点 id；root 没有父节点。
    parent: Option<NodeId>,
    /// 被多少个 active handle 保护。为 0 的 leaf 才允许被 evict。
    ref_count: usize,
    /// 稳定 id，用于 LRU heap 里 timestamp 相同时做 deterministic tie-break。
    uuid: usize,
    /// 访问时间戳，用于 leaf LRU eviction。
    timestamp: u64,
    /// 节点被 evict 后不会从 `nodes` Vec 里物理删除，只标记为 dead，避免 NodeId 失效。
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
    /// 创建一个空的 radix prefix cache。
    ///
    /// `page_size` 决定插入和匹配时的对齐粒度。所有插入长度都会向下对齐到 page 边界。
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

    /// 查找 `input_ids` 在 radix tree 里的最长已缓存前缀。
    ///
    /// 注意：这个方法不是纯读操作。它会更新 LRU timestamp，并且当查询在某个压缩边中间
    /// 停止匹配时，会 split 该节点。因此并发包装器里也必须把它放在互斥锁下执行。
    pub fn match_prefix(&mut self, input_ids: &[Token]) -> MatchResult {
        let (node, prefix_len) = self.tree_walk(input_ids);
        MatchResult {
            handle: RadixCacheHandle {
                cached_len: prefix_len,
                node,
            },
        }
    }

    /// 插入一个 prefix 及其对应的 KV cache indices。
    ///
    /// 返回值里的 `cached_len` 表示插入前已经存在的前缀长度；调用方如果管理外部 cache
    /// page，可以据此释放重复写入的部分。
    pub fn insert_prefix(&mut self, input_ids: &[Token], indices: &[Token]) -> InsertResult {
        assert_eq!(
            input_ids.len(),
            indices.len(),
            "input_ids/indices lengths must match"
        );
        // 只缓存完整 page，未对齐的尾部 token 不进入 prefix cache。
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

    /// 锁定一个 handle 对应路径，防止其已匹配节点被 eviction 删除。
    ///
    /// 锁定操作沿着 handle 节点一直向上更新 ref_count；当节点从 0 变为 1 时，
    /// 该节点从 evictable 区转入 protected 区。
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

    /// 释放一个之前锁定过的 handle。
    ///
    /// 当节点 ref_count 从 1 降为 0 时，它重新变为 evictable。
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

    /// 根据 handle 回溯到 root，拼接完整命中前缀对应的 KV cache indices。
    ///
    /// 调用方应保证 handle 在使用期间没有被 evict；并发场景下建议在 `with_cache` 里
    /// 完成 `match -> lock -> get -> unlock` 的原子序列。
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

    /// 从未被锁定的 leaf 节点中按 LRU 顺序驱逐至少 `size` 个 token。
    ///
    /// 返回被驱逐的 KV cache indices。由于 radix tree 以节点为粒度删除，实际驱逐长度
    /// 可能大于请求的 `size`。
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

    /// 沿 radix tree 查找最长 prefix。
    ///
    /// 返回 `(最后命中的节点, 命中的 token 长度)`。如果查询只匹配到某个压缩边的一部分，
    /// 会在 page 边界处 split 该边，让返回的节点正好代表已命中的 prefix。
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

            // 完整经过该节点时刷新 timestamp，作为 LRU eviction 的最近访问依据。
            self.nodes[node].timestamp = tic;
        }

        (node, prefix_len)
    }

    /// 将一个压缩边节点在 `pos` 处拆成父子两个节点。
    ///
    /// split 后：
    /// - 新节点保存原 key/value 的前半段，并接到原 parent 下；
    /// - 原节点保存后半段，并成为新节点的 child；
    /// - 原节点的 children 不变，因为它们仍属于后半段 prefix。
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

    /// 生成 children map 的索引 key。
    ///
    /// 正常情况下取一个完整 page；如果查询剩余长度小于 page_size，则使用剩余 token。
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

    /// 收集当前可驱逐的 leaf 节点，并放入按 timestamp 排序的最小堆语义结构。
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

    /// 递归校验 radix tree 不变量，用于测试和调试。
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

pub type SharedRadixPrefixCache = Arc<ConcurrentRadixPrefixCache>;

/// A coarse-grained concurrent wrapper around [`RadixPrefixCache`].
///
/// This is intentionally a single `Mutex` around the full tree. It matches the
/// current "single-threaded scheduler / external mutex serialization" model:
/// every operation is serialized, so callers can safely share the cache across
/// threads with `Arc<ConcurrentRadixPrefixCache>` without introducing data races.
///
/// For multi-step sequences that must be atomic, such as `match_prefix ->
/// lock_handle -> get_matched_indices`, prefer [`ConcurrentRadixPrefixCache::lock`]
/// or [`ConcurrentRadixPrefixCache::with_cache`] and perform the sequence while
/// holding the guard.
#[derive(Debug)]
pub struct ConcurrentRadixPrefixCache {
    inner: Mutex<RadixPrefixCache>,
}

impl ConcurrentRadixPrefixCache {
    /// 创建一个粗粒度互斥的并发 wrapper。
    pub fn new(page_size: usize) -> Self {
        Self {
            inner: Mutex::new(RadixPrefixCache::new(page_size)),
        }
    }

    /// 直接返回 `Arc` 包装后的共享 cache，方便跨线程 clone 和传递。
    pub fn shared(page_size: usize) -> SharedRadixPrefixCache {
        Arc::new(Self::new(page_size))
    }

    /// 将一个已有的单线程 cache 转换成并发安全 wrapper。
    pub fn from_cache(cache: RadixPrefixCache) -> Self {
        Self {
            inner: Mutex::new(cache),
        }
    }

    /// 获取底层 cache 的互斥 guard。
    ///
    /// 当调用方需要执行一组不可被其他线程插入的多步操作时使用该方法。
    pub fn lock(&self) -> MutexGuard<'_, RadixPrefixCache> {
        self.inner
            .lock()
            .expect("radix prefix cache mutex poisoned")
    }

    /// 在同一把锁下执行闭包。
    ///
    /// 这是推荐的多步原子操作入口，例如 `match -> lock_handle -> get_indices`。
    pub fn with_cache<R>(&self, f: impl FnOnce(&mut RadixPrefixCache) -> R) -> R {
        let mut cache = self.lock();
        f(&mut cache)
    }

    pub fn page_size(&self) -> usize {
        self.with_cache(|cache| cache.page_size())
    }

    pub fn size_info(&self) -> SizeInfo {
        self.with_cache(|cache| cache.size_info())
    }

    pub fn match_prefix(&self, input_ids: &[Token]) -> MatchResult {
        self.with_cache(|cache| cache.match_prefix(input_ids))
    }

    pub fn insert_prefix(&self, input_ids: &[Token], indices: &[Token]) -> InsertResult {
        self.with_cache(|cache| cache.insert_prefix(input_ids, indices))
    }

    pub fn lock_handle(&self, handle: RadixCacheHandle) {
        self.with_cache(|cache| cache.lock_handle(handle));
    }

    pub fn unlock_handle(&self, handle: RadixCacheHandle) {
        self.with_cache(|cache| cache.unlock_handle(handle));
    }

    pub fn get_matched_indices(&self, handle: RadixCacheHandle) -> Vec<Token> {
        self.with_cache(|cache| cache.get_matched_indices(handle))
    }

    pub fn evict(&self, size: usize) -> Vec<Token> {
        self.with_cache(|cache| cache.evict(size))
    }

    pub fn reset(&self) {
        self.with_cache(|cache| cache.reset());
    }

    pub fn check_integrity(&self) -> Result<(), String> {
        self.with_cache(|cache| cache.check_integrity())
    }

    pub fn debug_dump(&self) -> Vec<(usize, Option<usize>, Vec<Token>, Vec<Token>, usize)> {
        self.with_cache(|cache| cache.debug_dump())
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
    use std::thread;

    fn assert_send_sync<T: Send + Sync>() {}

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

    #[test]
    fn concurrent_wrapper_is_send_and_sync() {
        assert_send_sync::<ConcurrentRadixPrefixCache>();
        assert_send_sync::<SharedRadixPrefixCache>();
    }

    #[test]
    fn concurrent_wrapper_serializes_parallel_inserts() {
        let cache = ConcurrentRadixPrefixCache::shared(1);
        let mut workers = Vec::new();

        for worker_id in 0..8 {
            let cache = Arc::clone(&cache);
            workers.push(thread::spawn(move || {
                for local_id in 0..128 {
                    let base = worker_id * 10_000 + local_id * 4;
                    let input_ids = vec![base, base + 1, base + 2, base + 3];
                    let indices = vec![base * 10, base * 10 + 1, base * 10 + 2, base * 10 + 3];
                    cache.insert_prefix(&input_ids, &indices);
                }
            }));
        }

        for worker in workers {
            worker.join().unwrap();
        }

        assert_eq!(cache.size_info().total_size(), 8 * 128 * 4);
        cache.check_integrity().unwrap();

        let matched = cache.match_prefix(&[20_000, 20_001, 20_002, 20_003, 42]);
        assert_eq!(matched.handle.cached_len, 4);
        assert_eq!(
            cache.get_matched_indices(matched.handle),
            vec![200_000, 200_001, 200_002, 200_003]
        );
    }

    #[test]
    fn concurrent_wrapper_supports_atomic_multi_step_with_guard() {
        let cache = ConcurrentRadixPrefixCache::shared(1);
        cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]);

        let indices = cache.with_cache(|cache| {
            let matched = cache.match_prefix(&[1, 2, 3, 4]);
            cache.lock_handle(matched.handle);
            let indices = cache.get_matched_indices(matched.handle);
            cache.unlock_handle(matched.handle);
            indices
        });

        assert_eq!(indices, vec![10, 11, 12]);
        assert_eq!(cache.size_info().evictable_size, 3);
        cache.check_integrity().unwrap();
    }
}
