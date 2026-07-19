use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;

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
pub struct LookupResult {
    /// 命中的 prefix 长度。
    pub cached_len: usize,
    /// 命中 prefix 对应的 KV cache indices。
    pub indices: Vec<Token>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertResult {
    /// The length that was already present before this insertion.
    pub cached_len: usize,
    /// A handle pointing to the inserted / matched full aligned prefix.
    pub handle: RadixCacheHandle,
}

#[derive(Debug, Clone)]
struct TokenSpan {
    data: Arc<[Token]>,
    start: usize,
    len: usize,
}

impl TokenSpan {
    fn empty() -> Self {
        Self {
            data: Arc::from([]),
            start: 0,
            len: 0,
        }
    }

    fn from_slice(tokens: &[Token]) -> Self {
        Self {
            data: Arc::from(tokens),
            start: 0,
            len: tokens.len(),
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn as_slice(&self) -> &[Token] {
        &self.data[self.start..self.start + self.len]
    }

    fn prefix(&self, len: usize) -> Self {
        assert!(len <= self.len, "prefix length out of range");
        Self {
            data: Arc::clone(&self.data),
            start: self.start,
            len,
        }
    }

    fn suffix_from(&self, offset: usize) -> Self {
        assert!(offset <= self.len, "suffix offset out of range");
        Self {
            data: Arc::clone(&self.data),
            start: self.start + offset,
            len: self.len - offset,
        }
    }

    fn to_vec(&self) -> Vec<Token> {
        self.as_slice().to_vec()
    }
}

#[derive(Debug, Clone)]
struct RadixTreeNode {
    /// 该压缩边保存的 token 片段。
    key: TokenSpan,
    /// 与 `key` 一一对应的 KV cache index 片段。
    value: TokenSpan,
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
            key: TokenSpan::empty(),
            value: TokenSpan::empty(),
            children: HashMap::new(),
            parent: None,
            ref_count: 1,
            uuid: ROOT,
            timestamp,
            alive: true,
        }
    }

    fn new(key: TokenSpan, value: TokenSpan, parent: NodeId, uuid: usize, timestamp: u64) -> Self {
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

    /// 只读查找 `input_ids` 的最长已缓存前缀。
    ///
    /// 与 [`RadixPrefixCache::match_prefix`] 不同，这个方法不会更新 timestamp，也不会 split
    /// radix tree 节点，因此可以安全放在并发 wrapper 的读锁里执行。返回的 handle 可能指向
    /// 一个比 `cached_len` 更长的压缩边节点；调用 [`RadixPrefixCache::get_matched_indices`]
    /// 时会按照 `cached_len` 截断。
    pub fn lookup_prefix(&self, input_ids: &[Token]) -> MatchResult {
        let (node, prefix_len) = self.read_tree_walk(input_ids);
        MatchResult {
            handle: RadixCacheHandle {
                cached_len: prefix_len,
                node,
            },
        }
    }

    /// 只读查找并直接返回命中的 KV cache indices。
    ///
    /// 这是并发读场景更推荐的接口：查找和收集 indices 在同一个读锁生命周期内完成，避免
    /// 调用方拿到 handle 后又被其他写线程 evict 的竞态窗口。
    pub fn lookup_indices(&self, input_ids: &[Token]) -> LookupResult {
        let matched = self.lookup_prefix(input_ids);
        LookupResult {
            cached_len: matched.handle.cached_len,
            indices: self.get_matched_indices(matched.handle),
        }
    }

    /// 只读查找并将命中的 KV cache indices 写入调用方提供的 buffer。
    ///
    /// 该接口会先清空 `output`，然后复用其已有容量写入结果，适合高频 lookup 场景减少
    /// 每次返回 `Vec` 带来的 allocation。返回值为命中的 prefix 长度。
    pub fn lookup_indices_into(&self, input_ids: &[Token], output: &mut Vec<Token>) -> usize {
        let matched = self.lookup_prefix(input_ids);
        self.get_matched_indices_into(matched.handle, output);
        matched.handle.cached_len
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
            let new_node = self.add_child(node, &input_ids[prefix_len..], &indices[prefix_len..]);
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
        let mut result = Vec::with_capacity(handle.cached_len);
        self.append_matched_indices(handle, &mut result);
        result
    }

    /// 根据 handle 回溯到 root，将命中的 KV cache indices 写入调用方提供的 buffer。
    ///
    /// 相比 [`RadixPrefixCache::get_matched_indices`]，该接口可以复用 `output` 的容量，避免
    /// 热路径上反复分配新的 `Vec`。
    pub fn get_matched_indices_into(&self, handle: RadixCacheHandle, output: &mut Vec<Token>) {
        output.clear();
        output.reserve(handle.cached_len);
        self.append_matched_indices(handle, output);
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

        let mut evicted_indices = Vec::with_capacity(size);
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
            evicted_indices.extend_from_slice(self.nodes[node].value.as_slice());
            self.evictable_size -= self.nodes[node].len();

            let parent = self.nodes[node]
                .parent
                .expect("evicted node must not be root");
            let child_key = self.child_key_owned(self.nodes[node].key.as_slice());
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
                    n.key.to_vec(),
                    n.value.to_vec(),
                    n.ref_count,
                )
            })
            .collect()
    }

    fn add_child(&mut self, parent: NodeId, key: &[Token], value: &[Token]) -> NodeId {
        let child_key = self.child_key_owned(key);
        let node = self.nodes.len();
        let timestamp = self.tick();
        self.nodes.push(RadixTreeNode::new(
            TokenSpan::from_slice(key),
            TokenSpan::from_slice(value),
            parent,
            node,
            timestamp,
        ));
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
            let key = self.child_key_slice(&input_ids[prefix_len..]);
            let Some(child) = self.nodes[node].children.get(key).copied() else {
                return (node, prefix_len);
            };
            node = child;

            let match_len = align_down(
                common_prefix_len(self.nodes[node].key.as_slice(), &input_ids[prefix_len..]),
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

    /// 纯读版本的 tree walk。
    ///
    /// 它只计算最长 page-aligned prefix，不更新 timestamp，也不做 split。若查询只命中某个
    /// 压缩边的前半段，返回值中的 `node` 仍然是该压缩边节点，`prefix_len` 则记录实际命中长度。
    fn read_tree_walk(&self, input_ids: &[Token]) -> (NodeId, usize) {
        let mut prefix_len = 0;
        let total_len = input_ids.len();
        let mut node = ROOT;

        while prefix_len < total_len {
            let key = self.child_key_slice(&input_ids[prefix_len..]);
            let Some(child) = self.nodes[node].children.get(key).copied() else {
                return (node, prefix_len);
            };
            node = child;

            let match_len = align_down(
                common_prefix_len(self.nodes[node].key.as_slice(), &input_ids[prefix_len..]),
                self.page_size,
            );
            prefix_len += match_len;

            if match_len != self.nodes[node].len() {
                return (node, prefix_len);
            }
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

        let old_ref_count = self.nodes[node].ref_count;
        let old_child_key = self.child_key_owned(self.nodes[node].key.as_slice());

        let prefix_key = self.nodes[node].key.prefix(pos);
        let prefix_value = self.nodes[node].value.prefix(pos);
        self.nodes[node].key = self.nodes[node].key.suffix_from(pos);
        self.nodes[node].value = self.nodes[node].value.suffix_from(pos);

        let new_parent_id = self.nodes.len();
        let mut new_parent =
            RadixTreeNode::new(prefix_key, prefix_value, parent, new_parent_id, timestamp);
        new_parent.ref_count = old_ref_count;

        self.nodes.push(new_parent);
        self.nodes[parent]
            .children
            .insert(old_child_key, new_parent_id);

        self.nodes[node].parent = Some(new_parent_id);

        let new_child_key = self.child_key_owned(self.nodes[node].key.as_slice());
        self.nodes[new_parent_id]
            .children
            .insert(new_child_key, node);

        new_parent_id
    }

    /// 生成 children map 的索引 key。
    ///
    /// 正常情况下取一个完整 page；如果查询剩余长度小于 page_size，则使用剩余 token。
    fn child_key_slice<'a>(&self, tokens: &'a [Token]) -> &'a [Token] {
        assert!(!tokens.is_empty(), "child key requires at least one token");
        &tokens[..tokens.len().min(self.page_size)]
    }

    fn child_key_owned(&self, tokens: &[Token]) -> ChildKey {
        self.child_key_slice(tokens).to_vec()
    }

    fn append_matched_indices(&self, handle: RadixCacheHandle, output: &mut Vec<Token>) {
        self.append_node_indices_from_root(handle.node, handle.cached_len, output);
    }

    fn append_node_indices_from_root(
        &self,
        node: NodeId,
        remaining: usize,
        output: &mut Vec<Token>,
    ) -> usize {
        if node == ROOT || remaining == 0 {
            return remaining;
        }

        let parent = self.nodes[node]
            .parent
            .expect("non-root node must have parent");
        let remaining = self.append_node_indices_from_root(parent, remaining, output);
        if remaining == 0 {
            return 0;
        }

        let n = &self.nodes[node];
        assert!(n.alive, "handle points to an evicted node");
        let take = remaining.min(n.value.len());
        output.extend_from_slice(&n.value.as_slice()[..take]);
        remaining - take
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
            let expected_key = self.child_key_slice(self.nodes[*child].key.as_slice());
            if key.as_slice() != expected_key {
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
