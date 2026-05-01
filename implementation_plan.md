# Implementation Plan: Out-of-Core Unified DAG (V3)

Implement a high-performance, memory-efficient DAG engine that uses `mmap` for node storage on ZFS and a compact hash index in RAM. This will enable a full, unified global audit of TinyLlama-1.1B without memory constraints or algorithmic bottlenecks.

## User Review Required

> [!IMPORTANT]
> This plan requires creating a new ZFS dataset for high-IO mmap operations. Please ensure you have sufficient permissions for `sudo zfs create`.

> [!NOTE]
> We are adding `memmap2` as a dependency. This is the industry standard for safe memory mapping in Rust.

## Proposed Changes

### 1. Storage Layer (ZFS)
- Create a dedicated ZFS dataset `vectorlegis_ssd_pool/eml_working` with LZ4 compression and 128K recordsize for optimal mmap performance.

---

### 2. DAG Engine (Core)

#### [NEW] [dag_mmap.rs](file:///my_data/engine/src/dag_mmap.rs)
- **CompactNode**: A 24-byte POD (Plain Old Data) structure containing:
  - `tag` (u8), `left` (u32), `right` (u32), `hash` (u64), `ref_count` (u32).
- **MmapDag**:
  - `mmap`: `memmap2::MmapMut` for zero-copy disk access to nodes.
  - **Pre-allocation**: Start with a large `initial_capacity` (e.g., 200M nodes = 4.8GB). This avoids risky `grow()` operations and takes advantage of ZFS lazy block allocation.
  - `hash_index`: `HashMap<u64, u32>` (RAM) for fast global deduplication. Never cleared.
  - `var_names` & `const_values`: Interning for leaf nodes.

---

### 3. Integration & Benchmarking

#### [NEW] [full_layer_unified_v3.rs](file:///my_data/engine/src/bin/full_layer_unified_v3.rs)
- Implements the exhaustive audit using the `MmapDag`.
- **Key Algorithmic Change**: Uses `local_cache: HashMap<usize, u32>` created **fresh for each row** (or tree). This ensures $O(n)$ hashing within the tree while maintaining global deduplication via the persistent `MmapDag::hash_index`.

---

### 4. Dependencies

#### [MODIFY] [Cargo.toml](file:///my_data/engine/Cargo.toml)
- Add `memmap2 = "0.9"`.

## Verification Plan

### Automated Tests
1. **Unit Test for MmapDag**: 
   - Create a DAG, add nodes, flush, reopen, and verify structure.
   - Verify that adding identical trees results in zero new unique nodes.
3. **Cross-row Deduplication Test**:
   - Verify that shared `x_vars` (same `Arc` pointers used in different trees/rows) are correctly identified as the same node in the global DAG despite having separate local caches.

### Manual Verification
- Run `cargo run --bin full_layer_unified_v3 --release`.
- Monitor RAM usage (should stay around 1.5-2.0 GB).
- Monitor Disk IO (ZFS should show compression benefits).
- Verify the final `sharing_savings()` metric for the global model.
