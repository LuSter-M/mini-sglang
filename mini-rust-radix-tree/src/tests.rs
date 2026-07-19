use std::sync::mpsc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::*;

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
fn lookup_prefix_is_read_only_and_does_not_split() {
    let mut cache = RadixPrefixCache::new(1);
    cache.insert_prefix(&[1, 2, 3, 4], &[10, 11, 12, 13]);
    let before = cache.debug_dump().len();

    let lookup = cache.lookup_indices(&[1, 2, 9]);
    assert_eq!(lookup.cached_len, 2);
    assert_eq!(lookup.indices, vec![10, 11]);
    assert_eq!(cache.debug_dump().len(), before);

    let matched = cache.match_prefix(&[1, 2, 9]);
    assert_eq!(matched.handle.cached_len, 2);
    assert!(cache.debug_dump().len() > before);
    cache.check_integrity().unwrap();
}

#[test]
fn lookup_prefix_matches_full_prefix_without_mutation() {
    let mut cache = RadixPrefixCache::new(4);
    cache.insert_prefix(
        &[0, 1, 2, 3, 4, 5, 6, 7],
        &[100, 101, 102, 103, 104, 105, 106, 107],
    );

    let lookup = cache.lookup_indices(&[0, 1, 2, 3, 4, 5, 6, 7, 99]);
    assert_eq!(lookup.cached_len, 8);
    assert_eq!(lookup.indices, vec![100, 101, 102, 103, 104, 105, 106, 107]);
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
fn concurrent_wrapper_allows_parallel_read_lookups() {
    let cache = ConcurrentRadixPrefixCache::shared(1);
    for prefix_id in 0..128 {
        let base = prefix_id * 16;
        let input_ids: Vec<_> = (base..base + 16).collect();
        let indices: Vec<_> = (base * 10..base * 10 + 16).collect();
        cache.insert_prefix(&input_ids, &indices);
    }

    let mut workers = Vec::new();
    for worker_id in 0..8 {
        let cache = Arc::clone(&cache);
        workers.push(thread::spawn(move || {
            for round in 0..512 {
                let prefix_id = (worker_id * 17 + round) % 128;
                let base = prefix_id * 16;
                let mut query: Vec<_> = (base..base + 16).collect();
                query.push(-1);
                let lookup = cache.lookup_indices(&query);
                assert_eq!(lookup.cached_len, 16);
                assert_eq!(lookup.indices[0], base * 10);
            }
        }));
    }

    for worker in workers {
        worker.join().unwrap();
    }
    cache.check_integrity().unwrap();
}

#[test]
fn lookup_indices_into_reuses_output_buffer() {
    let mut cache = RadixPrefixCache::new(1);
    cache.insert_prefix(&[1, 2, 3, 4], &[10, 11, 12, 13]);

    let mut output = Vec::with_capacity(16);
    output.extend_from_slice(&[-1, -2, -3]);
    let original_capacity = output.capacity();

    let cached_len = cache.lookup_indices_into(&[1, 2, 3, 4, 5], &mut output);

    assert_eq!(cached_len, 4);
    assert_eq!(output, vec![10, 11, 12, 13]);
    assert_eq!(output.capacity(), original_capacity);
}

#[test]
fn concurrent_wrapper_lookup_indices_into_reuses_output_buffer() {
    let cache = ConcurrentRadixPrefixCache::shared(1);
    cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]);

    let mut output = Vec::with_capacity(8);
    let original_capacity = output.capacity();
    let cached_len = cache.lookup_indices_into(&[1, 2, 3, 4], &mut output);

    assert_eq!(cached_len, 3);
    assert_eq!(output, vec![10, 11, 12]);
    assert_eq!(output.capacity(), original_capacity);
}

#[test]
fn concurrent_wrapper_writer_waits_for_read_guard() {
    let cache = ConcurrentRadixPrefixCache::shared(1);
    cache.insert_prefix(&[1, 2, 3], &[10, 11, 12]);

    let read_guard = cache.read_lock();
    let writer_cache = Arc::clone(&cache);
    let (tx, rx) = mpsc::channel();

    let writer = thread::spawn(move || {
        writer_cache.insert_prefix(&[4, 5, 6], &[40, 41, 42]);
        tx.send(()).unwrap();
    });

    assert!(rx.recv_timeout(Duration::from_millis(30)).is_err());
    drop(read_guard);
    rx.recv_timeout(Duration::from_secs(1)).unwrap();
    writer.join().unwrap();

    assert_eq!(cache.lookup_prefix(&[4, 5, 6]).handle.cached_len, 3);
    cache.check_integrity().unwrap();
}

#[test]
fn concurrent_wrapper_handles_mixed_reads_writes_and_evictions() {
    let cache = ConcurrentRadixPrefixCache::shared(1);
    let mut protected_handles = Vec::new();

    for prefix_id in 0..64 {
        let base = prefix_id * 16;
        let input_ids: Vec<_> = (base..base + 16).collect();
        let indices: Vec<_> = (base * 10..base * 10 + 16).collect();
        let handle = cache.insert_prefix(&input_ids, &indices).handle;
        cache.lock_handle(handle);
        protected_handles.push(handle);
    }

    let mut workers = Vec::new();

    for worker_id in 0..4 {
        let cache = Arc::clone(&cache);
        workers.push(thread::spawn(move || {
            for round in 0..512 {
                let prefix_id = (worker_id * 13 + round) % 64;
                let base = prefix_id * 16;
                let mut query: Vec<_> = (base..base + 16).collect();
                query.push(-1);

                let lookup = cache.lookup_indices(&query);
                assert_eq!(lookup.cached_len, 16);
                assert_eq!(lookup.indices.len(), 16);
                assert_eq!(lookup.indices[0], base * 10);
            }
        }));
    }

    for writer_id in 0..2 {
        let cache = Arc::clone(&cache);
        workers.push(thread::spawn(move || {
            for local_id in 0..256 {
                let base = 100_000 + writer_id * 10_000 + local_id * 16;
                let input_ids: Vec<_> = (base..base + 16).collect();
                let indices: Vec<_> = (base * 10..base * 10 + 16).collect();
                cache.with_write_cache(|cache| {
                    let inserted = cache.insert_prefix(&input_ids, &indices);
                    cache.lock_handle(inserted.handle);
                    cache.unlock_handle(inserted.handle);
                });
            }
        }));
    }

    let evict_cache = Arc::clone(&cache);
    workers.push(thread::spawn(move || {
        for _ in 0..128 {
            if evict_cache.size_info().evictable_size >= 16 {
                evict_cache.evict(16);
            }
        }
    }));

    for worker in workers {
        worker.join().unwrap();
    }

    for handle in protected_handles {
        cache.unlock_handle(handle);
    }

    cache.check_integrity().unwrap();
    for prefix_id in 0..64 {
        let base = prefix_id * 16;
        assert_eq!(
            cache
                .lookup_prefix(&(base..base + 16).collect::<Vec<_>>())
                .handle
                .cached_len,
            16
        );
    }
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
