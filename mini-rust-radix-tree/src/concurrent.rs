use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::{
    InsertResult, LookupResult, MatchResult, RadixCacheHandle, RadixPrefixCache, SizeInfo, Token,
};

pub type SharedRadixPrefixCache = Arc<ConcurrentRadixPrefixCache>;

/// A read/write concurrent wrapper around [`RadixPrefixCache`].
///
/// The wrapper uses a single `RwLock` around the full tree:
/// - pure read APIs, such as [`ConcurrentRadixPrefixCache::lookup_prefix`] and
///   [`ConcurrentRadixPrefixCache::lookup_indices`], can run concurrently;
/// - mutating APIs, such as [`ConcurrentRadixPrefixCache::match_prefix`], insert,
///   lock/unlock and eviction, still take the exclusive write lock.
///
/// `match_prefix` is intentionally treated as a write operation because it updates
/// LRU timestamps and may split compressed radix-tree edges.
#[derive(Debug)]
pub struct ConcurrentRadixPrefixCache {
    inner: RwLock<RadixPrefixCache>,
}

impl ConcurrentRadixPrefixCache {
    /// 创建一个读写分离的并发 wrapper。
    pub fn new(page_size: usize) -> Self {
        Self {
            inner: RwLock::new(RadixPrefixCache::new(page_size)),
        }
    }

    /// 直接返回 `Arc` 包装后的共享 cache，方便跨线程 clone 和传递。
    pub fn shared(page_size: usize) -> SharedRadixPrefixCache {
        Arc::new(Self::new(page_size))
    }

    /// 将一个已有的单线程 cache 转换成并发安全 wrapper。
    pub fn from_cache(cache: RadixPrefixCache) -> Self {
        Self {
            inner: RwLock::new(cache),
        }
    }

    /// 获取底层 cache 的写 guard。
    ///
    /// 当调用方需要执行一组不可被其他线程插入的多步操作时使用该方法。
    pub fn lock(&self) -> RwLockWriteGuard<'_, RadixPrefixCache> {
        self.write_lock()
    }

    /// 获取底层 cache 的共享读 guard。
    pub fn read_lock(&self) -> RwLockReadGuard<'_, RadixPrefixCache> {
        self.inner
            .read()
            .expect("radix prefix cache rwlock poisoned")
    }

    /// 获取底层 cache 的独占写 guard。
    pub fn write_lock(&self) -> RwLockWriteGuard<'_, RadixPrefixCache> {
        self.inner
            .write()
            .expect("radix prefix cache rwlock poisoned")
    }

    /// 在同一把写锁下执行闭包。
    ///
    /// 这是推荐的多步原子操作入口，例如 `match -> lock_handle -> get_indices`。
    pub fn with_cache<R>(&self, f: impl FnOnce(&mut RadixPrefixCache) -> R) -> R {
        self.with_write_cache(f)
    }

    /// 在同一把读锁下执行闭包，多个读闭包可以并发执行。
    pub fn with_read_cache<R>(&self, f: impl FnOnce(&RadixPrefixCache) -> R) -> R {
        let cache = self.read_lock();
        f(&cache)
    }

    /// 在同一把写锁下执行闭包。
    pub fn with_write_cache<R>(&self, f: impl FnOnce(&mut RadixPrefixCache) -> R) -> R {
        let mut cache = self.write_lock();
        f(&mut cache)
    }

    pub fn page_size(&self) -> usize {
        self.with_read_cache(|cache| cache.page_size())
    }

    pub fn size_info(&self) -> SizeInfo {
        self.with_read_cache(|cache| cache.size_info())
    }

    /// 纯读 prefix lookup：不更新 LRU，也不 split 节点，因此可并发执行。
    pub fn lookup_prefix(&self, input_ids: &[Token]) -> MatchResult {
        self.with_read_cache(|cache| cache.lookup_prefix(input_ids))
    }

    /// 纯读 prefix lookup，并在同一个读锁内收集命中的 cache indices。
    pub fn lookup_indices(&self, input_ids: &[Token]) -> LookupResult {
        self.with_read_cache(|cache| cache.lookup_indices(input_ids))
    }

    /// 纯读 prefix lookup，并复用调用方提供的输出 buffer 以减少 allocation。
    pub fn lookup_indices_into(&self, input_ids: &[Token], output: &mut Vec<Token>) -> usize {
        self.with_read_cache(|cache| cache.lookup_indices_into(input_ids, output))
    }

    /// 写路径 prefix match：会更新 LRU，并可能 split 节点。
    pub fn match_prefix(&self, input_ids: &[Token]) -> MatchResult {
        self.with_write_cache(|cache| cache.match_prefix(input_ids))
    }

    pub fn insert_prefix(&self, input_ids: &[Token], indices: &[Token]) -> InsertResult {
        self.with_write_cache(|cache| cache.insert_prefix(input_ids, indices))
    }

    pub fn lock_handle(&self, handle: RadixCacheHandle) {
        self.with_write_cache(|cache| cache.lock_handle(handle));
    }

    pub fn unlock_handle(&self, handle: RadixCacheHandle) {
        self.with_write_cache(|cache| cache.unlock_handle(handle));
    }

    pub fn get_matched_indices(&self, handle: RadixCacheHandle) -> Vec<Token> {
        self.with_read_cache(|cache| cache.get_matched_indices(handle))
    }

    pub fn get_matched_indices_into(&self, handle: RadixCacheHandle, output: &mut Vec<Token>) {
        self.with_read_cache(|cache| cache.get_matched_indices_into(handle, output));
    }

    pub fn evict(&self, size: usize) -> Vec<Token> {
        self.with_write_cache(|cache| cache.evict(size))
    }

    pub fn reset(&self) {
        self.with_write_cache(|cache| cache.reset());
    }

    pub fn check_integrity(&self) -> Result<(), String> {
        self.with_read_cache(|cache| cache.check_integrity())
    }

    pub fn debug_dump(&self) -> Vec<(usize, Option<usize>, Vec<Token>, Vec<Token>, usize)> {
        self.with_read_cache(|cache| cache.debug_dump())
    }
}
