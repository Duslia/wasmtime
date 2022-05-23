use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::binemit::CodeOffset;
use crate::ir::{Function, Inst, SourceLoc, StackSlot};
use crate::machinst::isle::UnwindInst;
use crate::machinst::{MachBufferFinalized, MachCompileResult};
use crate::{MachCallSite, MachReloc, MachSrcLoc, MachStackMap, MachTrap, ValueLabelsRanges};
use cranelift_entity::EntityRef as _;

pub struct CacheKey(u64);

impl std::fmt::Display for CacheKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

pub trait KeyValueStore {
    fn get(&self, key: CacheKey) -> Option<Vec<u8>>;

    fn write(&self, key: CacheKey, val: Vec<u8>) -> bool;
}

pub struct TmpFileCacheStore;

impl KeyValueStore for TmpFileCacheStore {
    fn get(&self, key: CacheKey) -> Option<Vec<u8>> {
        std::fs::read(format!("/tmp/clif-{}.compiled", key)).ok()
    }

    fn write(&self, key: CacheKey, val: Vec<u8>) -> bool {
        std::fs::write(format!("/tmp/clif-{key}.compiled"), val).is_ok()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFunc {
    src: crate::alloc::string::String,
    // TODO add compilation parameters too
    compile_result: CacheableMachCompileResult,
}

pub(crate) struct CacheInput {
    src: String,
    key: CacheKey,
    srcloc_offset: SourceLoc,
}

/// Relative source location.
///
/// This can be used to recompute source locations independently of the other functions in the
/// project.
#[derive(serde::Serialize, serde::Deserialize)]
struct RelSourceLoc(u32);

impl RelSourceLoc {
    fn new(loc: SourceLoc, offset: SourceLoc) -> Self {
        Self(loc.bits() - offset.bits())
    }
    fn expand(&self, offset: SourceLoc) -> SourceLoc {
        SourceLoc::new(self.0 + offset.bits())
    }
}

// --
// Copies of data structures that use `RelSourceLoc` instead of `SourceLoc`.

#[derive(serde::Serialize, serde::Deserialize)]
pub struct CacheableMachSrcLoc {
    start: CodeOffset,
    end: CodeOffset,
    loc: RelSourceLoc,
}

impl CacheableMachSrcLoc {
    fn new(loc: &MachSrcLoc, offset: SourceLoc) -> Self {
        Self {
            start: loc.start,
            end: loc.end,
            loc: RelSourceLoc::new(loc.loc, offset),
        }
    }

    fn expand(self, offset: SourceLoc) -> MachSrcLoc {
        MachSrcLoc {
            start: self.start,
            end: self.end,
            loc: self.loc.expand(offset),
        }
    }
}

// --
// Our final data structure.

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheableMachBufferFinalized {
    data: SmallVec<[u8; 1024]>,
    relocs: SmallVec<[MachReloc; 16]>,
    traps: SmallVec<[MachTrap; 16]>,
    call_sites: SmallVec<[MachCallSite; 16]>,
    srclocs: SmallVec<[CacheableMachSrcLoc; 64]>,
    stack_maps: SmallVec<[MachStackMap; 8]>,
    unwind_info: SmallVec<[(CodeOffset, UnwindInst); 8]>,
}

impl CacheableMachBufferFinalized {
    fn new(mbf: &MachBufferFinalized, offset: SourceLoc) -> Self {
        Self {
            data: mbf.data.clone(),
            relocs: mbf.relocs().into_iter().cloned().collect(),
            traps: mbf.traps().into_iter().cloned().collect(),
            call_sites: mbf.call_sites().into_iter().cloned().collect(),
            srclocs: mbf
                .get_srclocs_sorted()
                .iter()
                .map(|loc| CacheableMachSrcLoc::new(loc, offset))
                .collect(),
            stack_maps: mbf.stack_maps.clone(),
            unwind_info: mbf.unwind_info.clone(),
        }
    }

    fn expand(self, offset: SourceLoc) -> MachBufferFinalized {
        MachBufferFinalized {
            data: self.data,
            relocs: self.relocs,
            traps: self.traps,
            call_sites: self.call_sites,
            srclocs: self
                .srclocs
                .into_iter()
                .map(|loc| loc.expand(offset))
                .collect(),
            stack_maps: self.stack_maps,
            unwind_info: self.unwind_info,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CacheableMachCompileResult {
    buffer: CacheableMachBufferFinalized,
    frame_size: u32,
    disasm: Option<String>,
    value_labels_ranges: ValueLabelsRanges,
    stackslot_offsets: PrimaryMap<StackSlot, u32>,
    bb_starts: Vec<CodeOffset>,
    bb_edges: Vec<(CodeOffset, CodeOffset)>,
}

impl CacheableMachCompileResult {
    fn new(mcr: &MachCompileResult, offset: SourceLoc) -> Self {
        Self {
            buffer: CacheableMachBufferFinalized::new(&mcr.buffer, offset),
            frame_size: mcr.frame_size,
            disasm: mcr.disasm.clone(),
            value_labels_ranges: mcr.value_labels_ranges.clone(),
            stackslot_offsets: mcr.stackslot_offsets.clone(),
            bb_starts: mcr.bb_starts.clone(),
            bb_edges: mcr.bb_edges.clone(),
        }
    }

    fn expand(self, offset: SourceLoc) -> MachCompileResult {
        MachCompileResult {
            buffer: self.buffer.expand(offset),
            frame_size: self.frame_size,
            disasm: self.disasm,
            value_labels_ranges: self.value_labels_ranges,
            stackslot_offsets: self.stackslot_offsets,
            bb_starts: self.bb_starts,
            bb_edges: self.bb_edges,
        }
    }
}

/// Try to load a precompiled `MachCompileResult` from the given cache store.
///
/// If it fails because there's an input mismatch or it wasn't present, returns the cache key to be
/// used to store the result of the compilation later in `store`.
pub(crate) fn try_load(
    cache_store: &dyn KeyValueStore,
    func: &mut Function,
) -> Result<MachCompileResult, CacheInput> {
    // Use the input string as the cache key.
    //
    // Temporarily remove source locations as they're very likely to change in every single
    // function.

    let annotations = std::mem::take(&mut func.srclocs);
    use crate::alloc::string::ToString;
    let src = func.to_string();
    func.srclocs = annotations;

    let hash = {
        use core::hash::{Hash as _, Hasher as _};
        let mut hasher = std::collections::hash_map::DefaultHasher::new(); // fixed keys for determinism
        src.hash(&mut hasher);
        hasher.finish()
    };

    let srcloc_offset = func
        .srclocs
        .get(Inst::new(0))
        .cloned()
        .unwrap_or(SourceLoc::new(0));

    if let Some(bytes) = cache_store.get(CacheKey(hash)) {
        match bincode::deserialize::<CachedFunc>(bytes.as_slice()) {
            Ok(result) => {
                if src == result.src {
                    return Ok(result.compile_result.expand(srcloc_offset));
                }
            }
            Err(err) => {
                log::debug!("Couldn't deserialize cache entry with key {hash:x}: {err}");
            }
        }
    }

    Err(CacheInput {
        src,
        key: CacheKey(hash),
        srcloc_offset,
    })
}

/// Stores a `MachCompileResult` in the given cache store for the given key.
pub(crate) fn store(
    cache_store: &dyn KeyValueStore,
    input: CacheInput,
    result: &MachCompileResult,
) -> bool {
    let cached = CachedFunc {
        src: input.src,
        compile_result: CacheableMachCompileResult::new(result, input.srcloc_offset),
    };
    let mut did_cache = false;
    if let Ok(bytes) = bincode::serialize(&cached) {
        if cache_store.write(input.key, bytes) {
            did_cache = true;
        }
    }
    did_cache
}
