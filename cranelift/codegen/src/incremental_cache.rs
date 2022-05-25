use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::binemit::CodeOffset;
use crate::ir::{ExternalName, Function, Inst, SourceLoc, StackSlot};
use crate::machinst::isle::UnwindInst;
use crate::machinst::{MachBufferFinalized, MachCompileResult};
use crate::HashMap;
use crate::{MachCallSite, MachReloc, MachSrcLoc, MachStackMap, MachTrap, ValueLabelsRanges};
use cranelift_entity::EntityRef as _;
use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;

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
    external_names: Vec<ExtName>,
}

/// Relative source location.
///
/// This can be used to recompute source locations independently of the other functions in the
/// project.
#[derive(serde::Serialize, serde::Deserialize)]
struct RelSourceLoc(u32);

impl RelSourceLoc {
    fn new(loc: SourceLoc, offset: SourceLoc) -> Self {
        if loc.is_default() {
            Self(loc.bits())
        } else {
            Self(loc.bits() - offset.bits())
        }
    }
    fn expand(&self, offset: SourceLoc) -> SourceLoc {
        if SourceLoc::new(self.0).is_default() || offset.is_default() {
            SourceLoc::default()
        } else {
            SourceLoc::new(self.0 + offset.bits())
        }
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
struct ExtName {
    namespace: u32,
    index: u32,
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
    external_names: Vec<ExtName>,
}

impl CacheableMachCompileResult {
    fn new(mcr: &MachCompileResult, external_names: Vec<ExtName>, offset: SourceLoc) -> Self {
        Self {
            buffer: CacheableMachBufferFinalized::new(&mcr.buffer, offset),
            frame_size: mcr.frame_size,
            disasm: mcr.disasm.clone(),
            value_labels_ranges: mcr.value_labels_ranges.clone(),
            stackslot_offsets: mcr.stackslot_offsets.clone(),
            bb_starts: mcr.bb_starts.clone(),
            bb_edges: mcr.bb_edges.clone(),
            external_names,
        }
    }

    fn expand(self, offset: SourceLoc, external_names: Vec<ExtName>) -> MachCompileResult {
        let mut buffer = self.buffer.expand(offset);

        // Construct a mapping from compiled- to restored- external names.
        let mut map = HashMap::new();

        assert_eq!(external_names.len(), self.external_names.len());
        for (prev, after) in self.external_names.into_iter().zip(external_names) {
            if map.insert((prev.namespace, prev.index), after).is_some() {
                panic!(
                    "duplicate entry in prev->new namespace for {};{}",
                    prev.namespace, prev.index
                );
            }
        }

        // Adjust external names in relocations.
        for reloc in buffer.relocs.iter_mut() {
            if let ExternalName::User {
                ref mut namespace,
                ref mut index,
            } = reloc.name
            {
                if let Some(after) = map.get(&(*namespace, *index)) {
                    *namespace = after.namespace;
                    *index = after.index;
                } else {
                    panic!("didn't find previous mention of {};{}", namespace, index);
                }
            }
        }

        MachCompileResult {
            buffer,
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
    // TODO: hash something custom

    // Temporarily remove source locations as they're very likely to change in every single
    // function.
    let annotations = std::mem::take(&mut func.srclocs);

    let name = if let ExternalName::User {
        ref mut namespace,
        ref mut index,
    } = &mut func.name
    {
        let res = Some((*namespace, *index));
        *namespace = 0;
        *index = 0;
        res
    } else {
        None
    };

    // Temporarily remove any `ExternalName` that's called through a `call` (or equivalent) opcode:
    // - TODO in ExtFuncData (in func.dfg.ext_funcs)
    let external_names = {
        let mut names = Vec::with_capacity(func.dfg.ext_funcs.len()); // likely a short overestimate, but fine
        for func in func.dfg.ext_funcs.values_mut() {
            if let ExternalName::User {
                ref mut namespace,
                ref mut index,
            } = func.name
            {
                names.push(ExtName {
                    namespace: *namespace,
                    index: *index,
                });
                *namespace = 0;
                *index = 0;
            }
        }
        names
    };

    use crate::alloc::string::ToString;
    let src = func.to_string();

    // Restore source locations.
    func.srclocs = annotations;

    // Restore function name.
    if let Some((namespace, index)) = name {
        func.name = ExternalName::User { namespace, index };
    }

    // Restore function names.
    for (func, original_name) in func.dfg.ext_funcs.values_mut().zip(&external_names) {
        if let ExternalName::User {
            ref mut namespace,
            ref mut index,
        } = func.name
        {
            *namespace = original_name.namespace;
            *index = original_name.index;
        }
    }

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
                if src == result.src
                    && external_names.len() == result.compile_result.external_names.len()
                {
                    let mach_compile_result =
                        result.compile_result.expand(srcloc_offset, external_names);
                    return Ok(mach_compile_result);
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
        external_names,
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
        compile_result: CacheableMachCompileResult::new(
            result,
            input.external_names,
            input.srcloc_offset,
        ),
    };
    let mut did_cache = false;
    if let Ok(bytes) = bincode::serialize(&cached) {
        if cache_store.write(input.key, bytes) {
            did_cache = true;
        }
    }
    did_cache
}
