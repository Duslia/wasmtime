use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::binemit::CodeOffset;
use crate::ir::dfg::{BlockData, ValueData};
use crate::ir::function::VersionMarker;
use crate::ir::instructions::InstructionData;
use crate::ir::{
    self, Block, Constant, ConstantData, ExternalName, FuncRef, Function, Immediate, Inst,
    JumpTables, Layout, LibCall, SigRef, Signature, SourceLoc, StackSlot, StackSlots, Value,
    ValueLabel, ValueLabelAssignments, TESTCASE_NAME_LENGTH,
};
use crate::machinst::isle::UnwindInst;
use crate::machinst::{MachBufferFinalized, MachCompileResult};
use crate::HashMap;
use crate::{MachCallSite, MachReloc, MachSrcLoc, MachStackMap, MachTrap, ValueLabelsRanges};
use alloc::collections::BTreeMap;
use cranelift_entity::PrimaryMap;
use cranelift_entity::{EntityRef as _, SecondaryMap};
use smallvec::SmallVec;

pub struct CacheHashKey(u64);

impl std::fmt::Display for CacheHashKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

pub trait KeyValueStore {
    fn get(&self, key: CacheHashKey) -> Option<Vec<u8>>;

    fn write(&self, key: CacheHashKey, val: Vec<u8>) -> bool;
}

pub struct TmpFileCacheStore;

impl KeyValueStore for TmpFileCacheStore {
    fn get(&self, key: CacheHashKey) -> Option<Vec<u8>> {
        std::fs::read(format!("/tmp/clif-{}.compiled", key)).ok()
    }

    fn write(&self, key: CacheHashKey, val: Vec<u8>) -> bool {
        std::fs::write(format!("/tmp/clif-{key}.compiled"), val).is_ok()
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFunc {
    src: CacheKey,
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
pub enum CachedExternalName {
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

#[derive(PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CachedDataFlowGraph {
    // --
    // Those fields are the same as in DataFlowGraph
    // --
    insts: PrimaryMap<Inst, InstructionData>,
    results: SecondaryMap<Inst, Vec<Value>>,
    blocks: PrimaryMap<Block, BlockData>,
    values: PrimaryMap<Value, ValueData>,
    signatures: PrimaryMap<SigRef, Signature>,
    old_signatures: SecondaryMap<SigRef, Option<Signature>>,
    immediates: PrimaryMap<Immediate, ConstantData>,

    // --
    // Fields that we tweaked for caching
    // --
    constants: CachedConstantPool,
    values_labels: Option<BTreeMap<Value, CachedValueLabelAssignments>>,
    ext_funcs: PrimaryMap<FuncRef, CachedExtFuncData>,
}

#[derive(PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CacheKey {
    version_marker: VersionMarker,
    signature: Signature,
    stack_slots: StackSlots,
    global_values: PrimaryMap<ir::GlobalValue, ir::GlobalValueData>,
    heaps: PrimaryMap<ir::Heap, ir::HeapData>,
    tables: PrimaryMap<ir::Table, ir::TableData>,
    jump_tables: JumpTables,
    dfg: CachedDataFlowGraph,
    layout: Layout,
    stack_limit: Option<ir::GlobalValue>,
}

impl CacheKey {
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
            .iter()
            .map(|(_reff, ext_data)| {
                let name = match ext_data.name {
                    ExternalName::User { .. } => CachedExternalName::User,
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
            stack_slots: f.stack_slots.clone(),
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

pub(crate) struct CacheInput {
    src: CacheKey,
    hash_key: CacheHashKey,
    srcloc_offset: SourceLoc,
    external_names: Vec<ExtName>,
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
pub struct CachedMachSrcLoc {
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

#[derive(serde::Serialize, serde::Deserialize)]
struct ExtName {
    namespace: u32,
    index: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedMachCompiledResult {
    buffer: CachedMachBufferFinalized,
    frame_size: u32,
    disasm: Option<String>,
    value_labels_ranges: ValueLabelsRanges,
    stackslot_offsets: PrimaryMap<StackSlot, u32>,
    bb_starts: Vec<CodeOffset>,
    bb_edges: Vec<(CodeOffset, CodeOffset)>,
    external_names: Vec<ExtName>,
}

impl CachedMachCompiledResult {
    fn new(mcr: &MachCompileResult, external_names: Vec<ExtName>, offset: SourceLoc) -> Self {
        Self {
            buffer: CachedMachBufferFinalized::new(&mcr.buffer, offset),
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
    let external_names = func
        .dfg
        .ext_funcs
        .values()
        .filter_map(|data| {
            if let ExternalName::User { namespace, index } = &data.name {
                Some(ExtName {
                    namespace: *namespace,
                    index: *index,
                })
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let srcloc_offset = func
        .srclocs
        .get(Inst::new(0))
        .cloned()
        .unwrap_or(SourceLoc::new(0));

    let src = CacheKey::new(func, srcloc_offset);

    let hash = {
        use core::hash::{Hash as _, Hasher as _};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        src.hash(&mut hasher);
        hasher.finish()
    };

    if let Some(bytes) = cache_store.get(CacheHashKey(hash)) {
        match bincode::deserialize::<CachedFunc>(bytes.as_slice()) {
            Ok(mut result) => {
                // Make sure the blocks and instructions are sequenced the same way as we might
                // have serialized them earlier. This is the symmetric of what's done in
                // `CacheKey`'s ctor.
                result.src.layout.full_renumber();

                if src == result.src {
                    if external_names.len() == result.compile_result.external_names.len() {
                        let mach_compile_result =
                            result.compile_result.expand(srcloc_offset, external_names);
                        return Ok(mach_compile_result);
                    }
                    eprintln!("{} not read from cache: external names mismatch", func.name);
                } else {
                    eprintln!("{} not read from cache: source mismatch", func.name);

                    //if src.version_marker != result.src.version_marker {
                        //eprintln!("     because of version marker")
                    //}
                    //if src.signature != result.src.signature {
                        //eprintln!("     because of signature")
                    //}
                    //if src.stack_slots != result.src.stack_slots {
                        //eprintln!("     because of stack slots")
                    //}
                    //if src.global_values != result.src.global_values {
                        //eprintln!("     because of global values")
                    //}
                    //if src.heaps != result.src.heaps {
                        //eprintln!("     because of heaps")
                    //}
                    //if src.tables != result.src.tables {
                        //eprintln!("     because of tables")
                    //}
                    //if src.jump_tables != result.src.jump_tables {
                        //eprintln!("     because of jump tables")
                    //}
                    //if src.dfg != result.src.dfg {
                        //eprintln!("     because of dfg")
                    //}
                    //if src.layout != result.src.layout {
                        //if func.layout.blocks().count() < 8 {
                            //eprintln!(
                                //"     because of layout:\n{:?}\n{:?}",
                                //src.layout, result.src.layout
                            //);
                        //} else {
                            //eprintln!("     because of layout",);
                        //}
                    //}
                    //if src.stack_limit != result.src.stack_limit {
                        //eprintln!("     because of stack limit")
                    //}
                }
            }
            Err(err) => {
                eprintln!("Couldn't deserialize cache entry with key {hash:x}: {err}");
                //log::debug!("Couldn't deserialize cache entry with key {hash:x}: {err}");
            }
        }
    } else {
        eprintln!(
            //log::trace!(
            "{} not read from cache: function hash {hash:x} not found",
            func.name
        );
    }

    Err(CacheInput {
        src,
        hash_key: CacheHashKey(hash),
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
        compile_result: CachedMachCompiledResult::new(
            result,
            input.external_names,
            input.srcloc_offset,
        ),
    };
    let mut did_cache = false;
    if let Ok(bytes) = bincode::serialize(&cached) {
        if cache_store.write(input.hash_key, bytes) {
            did_cache = true;
        }
    }
    did_cache
}
