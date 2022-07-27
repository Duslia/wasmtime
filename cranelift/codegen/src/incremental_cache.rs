//! TODO

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::binemit::CodeOffset;
use crate::ir::dfg::{BlockData, ValueDataPacked};
use crate::ir::function::VersionMarker;
use crate::ir::instructions::InstructionData;
use crate::ir::{
    self, Block, Constant, ConstantData, DynamicStackSlot, DynamicStackSlots, ExternalName,
    FuncRef, Function, Immediate, Inst, JumpTables, Layout, LibCall, SigRef, Signature, SourceLoc,
    StackSlot, StackSlots, Value, ValueLabel, ValueLabelAssignments, TESTCASE_NAME_LENGTH,
};
use crate::machinst::isle::UnwindInst;
use crate::machinst::{MachBufferFinalized, MachCompileResult};
use crate::HashMap;
use crate::{MachCallSite, MachReloc, MachSrcLoc, MachStackMap, MachTrap, ValueLabelsRanges};
use alloc::collections::BTreeMap;
use cranelift_entity::PrimaryMap;
use cranelift_entity::{EntityRef as _, SecondaryMap};
use smallvec::SmallVec;

/// Backing storage for the incremental compilation cache, when enabled.
#[cfg(feature = "incremental-cache")]
pub trait CacheStore {
    /// Given a cache key hash, retrieves the associated opaque serialized data.
    fn get(&self, key: CacheKeyHash) -> Option<&[u8]>;

    /// Given a new cache key and a serialized blob obtained from `serialize_compiled`, stores it
    /// in the cache store.
    ///
    /// Returns true when insertion is successful, false otherwise.
    fn insert(&mut self, key: CacheKeyHash, val: Vec<u8>) -> bool;
}

/// Hashed `CachedKey`, to use as an identifier when looking up whether a function has already been
/// compiled or not.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct CacheKeyHash(u64);

impl std::fmt::Display for CacheKeyHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFunc {
    cache_key: CacheKey,
    // TODO add compilation parameters too
    compile_result: CachedMachCompiledResult,
}

/// Same as `ValueLabelStart`, but we relocate the `from` source location.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Hash)]
struct CachedValueLabelStart {
    from: RelSourceLoc,
    label: ValueLabel,
}

/// Same as `ValueLabelAssignments`, but we relocate the `from` source location.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Hash)]
enum CachedValueLabelAssignments {
    Starts(alloc::vec::Vec<CachedValueLabelStart>),
    Alias { from: RelSourceLoc, value: Value },
}

#[derive(Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CachedConstantPool {
    handles_to_values: BTreeMap<Constant, ConstantData>,
    values_to_handles: BTreeMap<ConstantData, Constant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
enum CachedExternalName {
    User,
    TestCase {
        length: u8,
        ascii: [u8; TESTCASE_NAME_LENGTH],
    },
    LibCall(LibCall),
}

#[derive(Clone, Debug, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CachedExtFuncData {
    name: CachedExternalName,
    signature: SigRef,
    colocated: bool,
}

#[derive(Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CachedDataFlowGraph {
    // --
    // Those fields are the same as in DataFlowGraph
    // --
    insts: PrimaryMap<Inst, InstructionData>,
    results: SecondaryMap<Inst, Vec<Value>>,
    blocks: PrimaryMap<Block, BlockData>,
    values: PrimaryMap<Value, ValueDataPacked>,
    signatures: PrimaryMap<SigRef, Signature>,
    old_signatures: SecondaryMap<SigRef, Option<Signature>>,
    immediates: PrimaryMap<Immediate, ConstantData>,

    // --
    // Fields that we tweaked for caching
    // --
    constants: CachedConstantPool,
    values_labels: Option<BTreeMap<Value, CachedValueLabelAssignments>>,
    /// Same as `DataFlowGraph::ext_funcs`, but we remove the identifying fields of the
    /// `ExternalName` so two calls to external user functions appear the same and end up in the
    /// same cache bucket.
    ext_funcs: PrimaryMap<FuncRef, CachedExtFuncData>,
}

/// Key for caching a single function's compilation.
///
/// If two functions get the same `CacheKey`, then we can reuse the compiled artifacts, modulo some
/// relocation fixups.
#[derive(Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CacheKey {
    version_marker: VersionMarker,
    signature: Signature,
    sized_stack_slots: StackSlots,
    dynamic_stack_slots: DynamicStackSlots,
    global_values: PrimaryMap<ir::GlobalValue, ir::GlobalValueData>,
    heaps: PrimaryMap<ir::Heap, ir::HeapData>,
    tables: PrimaryMap<ir::Table, ir::TableData>,
    jump_tables: JumpTables,
    dfg: CachedDataFlowGraph,
    layout: Layout,
    stack_limit: Option<ir::GlobalValue>,
}

impl CacheKey {
    /// Creates a new cache store key for a function.
    ///
    /// This is a bit expensive to compute, so it should be cached and reused as much as possible.
    fn new(f: &Function, offset: SourceLoc) -> Self {
        let constants = CachedConstantPool {
            handles_to_values: f.dfg.constants.handles_to_values.clone(),
            values_to_handles: f
                .dfg
                .constants
                .values_to_handles
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        };

        let values_labels = f.dfg.values_labels.clone().map(|vl| {
            vl.into_iter()
                .map(|(k, v)| {
                    let v = match v {
                        ValueLabelAssignments::Starts(vec) => CachedValueLabelAssignments::Starts(
                            vec.into_iter()
                                .map(|vls| CachedValueLabelStart {
                                    from: RelSourceLoc::new(vls.from, offset),
                                    label: vls.label,
                                })
                                .collect(),
                        ),
                        ValueLabelAssignments::Alias { from, value } => {
                            CachedValueLabelAssignments::Alias {
                                from: RelSourceLoc::new(from, offset),
                                value,
                            }
                        }
                    };
                    (k, v)
                })
                .collect()
        });

        let ext_funcs = f
            .dfg
            .ext_funcs
            .values()
            .map(|ext_data| {
                let name = match ext_data.name {
                    ExternalName::User { .. } => {
                        // Remove the identifying properties of the call to that external function,
                        // as we want two functions with calls to different functions to reuse each
                        // other's cached artifacts.
                        CachedExternalName::User
                    }
                    ExternalName::TestCase { length, ascii } => {
                        CachedExternalName::TestCase { length, ascii }
                    }
                    ExternalName::LibCall(libcall) => CachedExternalName::LibCall(libcall),
                };
                let data = CachedExtFuncData {
                    name,
                    signature: ext_data.signature.clone(),
                    colocated: ext_data.colocated,
                };
                // Note: we rely on the iteration order being the same as the insertion order in
                // PrimaryMap, here.
                data
            })
            .collect();

        let mut results = SecondaryMap::with_capacity(f.dfg.results.capacity());
        let value_list = &f.dfg.value_lists;
        for (inst, values) in f.dfg.results.iter() {
            results[inst] = values.as_slice(value_list).to_vec();
        }

        let dfg = CachedDataFlowGraph {
            insts: f.dfg.insts.clone(),
            results,
            blocks: f.dfg.blocks.clone(),
            values: f.dfg.values.clone(),
            signatures: f.dfg.signatures.clone(),
            old_signatures: f.dfg.old_signatures.clone(),
            immediates: f.dfg.immediates.clone(),

            constants,
            values_labels,
            ext_funcs,
        };

        let mut layout = f.layout.clone();
        // Make sure the blocks and instructions are sequenced the same way as we might
        // have serialized them earlier. This is the symmetric of what's done in
        // `try_load`.
        layout.full_renumber();

        CacheKey {
            version_marker: f.version_marker,
            signature: f.signature.clone(),
            sized_stack_slots: f.sized_stack_slots.clone(),
            dynamic_stack_slots: f.dynamic_stack_slots.clone(),
            global_values: f.global_values.clone(),
            heaps: f.heaps.clone(),
            tables: f.tables.clone(),
            jump_tables: f.jump_tables.clone(),
            dfg,
            layout,
            stack_limit: f.stack_limit.clone(),
        }
    }
}

/// Relative source location.
///
/// This can be used to recompute source locations independently of the other functions in the
/// project.
#[derive(Debug, Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct RelSourceLoc(u32);

impl RelSourceLoc {
    fn new(loc: SourceLoc, offset: SourceLoc) -> Self {
        if loc.is_default() || offset.is_default() {
            Self(SourceLoc::default().bits())
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
struct CachedMachSrcLoc {
    start: CodeOffset,
    end: CodeOffset,
    loc: RelSourceLoc,
}

impl CachedMachSrcLoc {
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
struct CachedMachBufferFinalized {
    data: SmallVec<[u8; 1024]>,
    relocs: SmallVec<[MachReloc; 16]>,
    traps: SmallVec<[MachTrap; 16]>,
    call_sites: SmallVec<[MachCallSite; 16]>,
    srclocs: SmallVec<[CachedMachSrcLoc; 64]>,
    stack_maps: SmallVec<[MachStackMap; 8]>,
    unwind_info: SmallVec<[(CodeOffset, UnwindInst); 8]>,
}

impl CachedMachBufferFinalized {
    fn new(mbf: &MachBufferFinalized, offset: SourceLoc) -> Self {
        Self {
            data: mbf.data.clone(),
            relocs: mbf.relocs().into_iter().cloned().collect(),
            traps: mbf.traps().into_iter().cloned().collect(),
            call_sites: mbf.call_sites().into_iter().cloned().collect(),
            srclocs: mbf
                .get_srclocs_sorted()
                .iter()
                .map(|loc| CachedMachSrcLoc::new(loc, offset))
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

#[derive(Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct UserExternalName {
    namespace: u32,
    index: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedMachCompiledResult {
    buffer: CachedMachBufferFinalized,
    frame_size: u32,
    disasm: Option<String>,
    value_labels_ranges: ValueLabelsRanges,
    sized_stackslot_offsets: PrimaryMap<StackSlot, u32>,
    dynamic_stackslot_offsets: PrimaryMap<DynamicStackSlot, u32>,
    bb_starts: Vec<CodeOffset>,
    bb_edges: Vec<(CodeOffset, CodeOffset)>,

    /// Inverted mapping of user external name to `FuncRef`, constructed once to allow patching.
    func_refs: HashMap<UserExternalName, FuncRef>,
}

impl CachedMachCompiledResult {
    fn new(func: &Function, mcr: &MachCompileResult, offset: SourceLoc) -> Self {
        let func_refs = func
            .dfg
            .ext_funcs
            .iter()
            .filter_map(|(func_ref, ext_data)| {
                if let ExternalName::User { namespace, index } = &ext_data.name {
                    Some((
                        UserExternalName {
                            namespace: *namespace,
                            index: *index,
                        },
                        func_ref,
                    ))
                } else {
                    None
                }
            })
            .collect();

        Self {
            buffer: CachedMachBufferFinalized::new(&mcr.buffer, offset),
            frame_size: mcr.frame_size,
            disasm: mcr.disasm.clone(),
            value_labels_ranges: mcr.value_labels_ranges.clone(),
            sized_stackslot_offsets: mcr.sized_stackslot_offsets.clone(),
            dynamic_stackslot_offsets: mcr.dynamic_stackslot_offsets.clone(),
            bb_starts: mcr.bb_starts.clone(),
            bb_edges: mcr.bb_edges.clone(),
            func_refs,
        }
    }

    fn expand(self, after: &Function, offset: SourceLoc) -> MachCompileResult {
        let mut buffer = self.buffer.expand(offset);

        // Adjust external names in relocations.
        for reloc in buffer.relocs.iter_mut() {
            if let ExternalName::User {
                ref mut namespace,
                ref mut index,
            } = reloc.name
            {
                if let Some(func_ref) = self.func_refs.get(&UserExternalName {
                    namespace: *namespace,
                    index: *index,
                }) {
                    if let ExternalName::User {
                        namespace: new_namespace,
                        index: new_index,
                    } = &after.dfg.ext_funcs[*func_ref].name
                    {
                        *namespace = *new_namespace;
                        *index = *new_index;
                    } else {
                        panic!("unexpected kind of relocation??");
                    }
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
            sized_stackslot_offsets: self.sized_stackslot_offsets,
            dynamic_stackslot_offsets: self.dynamic_stackslot_offsets,
            bb_starts: self.bb_starts,
            bb_edges: self.bb_edges,
        }
    }
}

/// Compute a cache key, and hash it on your behalf.
///
/// Since computing the `CacheKey` is a bit expensive, it should be done as least as possible.
pub fn get_cache_key(func: &Function) -> (CacheKey, CacheKeyHash) {
    let srcloc_offset = func
        .srclocs
        .get(Inst::new(0))
        .cloned()
        .unwrap_or(SourceLoc::new(0));

    let cache_key = CacheKey::new(func, srcloc_offset);

    let hash = {
        use core::hash::{Hash as _, Hasher as _};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        cache_key.hash(&mut hasher);
        hasher.finish()
    };

    (cache_key, CacheKeyHash(hash))
}

/// Given a function that's been successfully compiled, serialize it to a blob that the caller may
/// store somewhere for future use by `try_finish_recompile`.
pub fn serialize_compiled(
    cache_key: CacheKey,
    func: &Function,
    result: &MachCompileResult,
) -> Result<Vec<u8>, bincode::Error> {
    let srcloc_offset = func
        .srclocs
        .get(Inst::new(0))
        .cloned()
        .unwrap_or(SourceLoc::new(0));

    let cached = CachedFunc {
        cache_key,
        compile_result: CachedMachCompiledResult::new(func, result, srcloc_offset),
    };

    bincode::serialize(&cached)
}

// TODO could the error return an indication why something went wrong?
/// Given a function that's been precompiled and its entry in the caching storage, try to shortcut
/// compilation of the given function.
pub fn try_finish_recompile(
    cache_key: &CacheKey,
    func: &Function,
    bytes: &[u8],
) -> Result<MachCompileResult, ()> {
    let srcloc_offset = func
        .srclocs
        .get(Inst::new(0))
        .cloned()
        .unwrap_or(SourceLoc::new(0));

    // try to deserialize, if not failure, return final recompiled code
    match bincode::deserialize::<CachedFunc>(bytes) {
        Ok(mut result) => {
            // Make sure the blocks and instructions are sequenced the same way as we might
            // have serialized them earlier. This is the symmetric of what's done in
            // `CacheKey`'s ctor.
            result.cache_key.layout.full_renumber();

            if *cache_key == result.cache_key {
                let mach_compile_result = result.compile_result.expand(func, srcloc_offset);
                return Ok(mach_compile_result);
            } else {
                eprintln!("{} not read from cache: source mismatch", func.name);

                //if cache_key.version_marker != result.cache_key.version_marker {
                //eprintln!("     because of version marker")
                //}
                //if cache_key.signature != result.cache_key.signature {
                //eprintln!("     because of signature")
                //}
                //if cache_key.stack_slots != result.cache_key.stack_slots {
                //eprintln!("     because of stack slots")
                //}
                //if cache_key.global_values != result.cache_key.global_values {
                //eprintln!("     because of global values")
                //}
                //if cache_key.heaps != result.cache_key.heaps {
                //eprintln!("     because of heaps")
                //}
                //if cache_key.tables != result.cache_key.tables {
                //eprintln!("     because of tables")
                //}
                //if cache_key.jump_tables != result.cache_key.jump_tables {
                //eprintln!("     because of jump tables")
                //}
                //if cache_key.dfg != result.cache_key.dfg {
                //eprintln!("     because of dfg")
                //}
                //if cache_key.layout != result.cache_key.layout {
                //if func.layout.blocks().count() < 8 {
                //eprintln!(
                //"     because of layout:\n{:?}\n{:?}",
                //cache_key.layout, result.cache_key.layout
                //);
                //} else {
                //eprintln!("     because of layout",);
                //}
                //}
                //if cache_key.stack_limit != result.cache_key.stack_limit {
                //eprintln!("     because of stack limit")
                //}
            }
        }

        Err(err) => {
            eprintln!("Couldn't deserialize cache entry: {err}");
            //log::debug!("Couldn't deserialize cache entry with key {hash:x}: {err}");
        }
    }

    Err(())
}
