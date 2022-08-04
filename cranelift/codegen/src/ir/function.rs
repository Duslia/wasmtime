//! Intermediate representation of a function.
//!
//! The `Function` struct defined in this module owns all of its basic blocks and
//! instructions.

use crate::entity::{PrimaryMap, SecondaryMap};
use crate::ir;
use crate::ir::JumpTables;
use crate::ir::{
    instructions::BranchInfo, Block, DynamicStackSlot, DynamicStackSlotData, DynamicType,
    ExtFuncData, FuncRef, GlobalValue, GlobalValueData, Heap, HeapData, Inst, InstructionData,
    JumpTable, JumpTableData, Opcode, SigRef, StackSlot, StackSlotData, Table, TableData, Type,
};
use crate::ir::{DataFlowGraph, ExternalName, Layout, Signature};
use crate::ir::{DynamicStackSlots, SourceLocs, StackSlots};
use crate::isa::CallConv;
use crate::value_label::ValueLabelsRanges;
use crate::write::write_function;
#[cfg(feature = "enable-serde")]
use alloc::string::String;
use core::fmt;

#[cfg(feature = "enable-serde")]
use serde::de::{Deserializer, Error};
#[cfg(feature = "enable-serde")]
use serde::ser::Serializer;
#[cfg(feature = "enable-serde")]
use serde::{Deserialize, Serialize};

use super::entities::UserExternalNameRef;
use super::extname::UserExternalName;
use super::{RelSourceLoc, SourceLoc};

/// A version marker used to ensure that serialized clif ir is never deserialized with a
/// different version of Cranelift.
#[derive(Copy, Clone, Debug, PartialEq, Hash)]
pub struct VersionMarker;

#[cfg(feature = "enable-serde")]
impl Serialize for VersionMarker {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        crate::VERSION.serialize(serializer)
    }
}

#[cfg(feature = "enable-serde")]
impl<'de> Deserialize<'de> for VersionMarker {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let version = String::deserialize(deserializer)?;
        if version != crate::VERSION {
            return Err(D::Error::custom(&format!(
                "Expected a clif ir function for version {}, found one for version {}",
                crate::VERSION,
                version,
            )));
        }
        Ok(VersionMarker)
    }
}

/// Function parameters used when creating this function, and that will become applied after
/// compilation to materialize the final `CompiledCode`.
#[derive(Clone)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct FunctionParameters {
    /// Name of this function. Mostly used by `.clif` files.
    pub name: ExternalName,

    /// The first `SourceLoc` appearing in the function, serving as a base for every relative
    /// source loc in the function.
    base_srcloc: Option<SourceLoc>,

    /// External user-defined function references.
    pub user_named_funcs: PrimaryMap<UserExternalNameRef, UserExternalName>,
}

impl FunctionParameters {
    /// Creates a new `FunctionParameters` with the given name.
    pub fn new(name: ExternalName) -> Self {
        Self {
            name,
            base_srcloc: None,
            user_named_funcs: Default::default(),
        }
    }

    /// Returns the base `SourceLoc`.
    ///
    /// Panics if the base source location was never set.
    pub fn base_srcloc(&self) -> SourceLoc {
        self.base_srcloc
            .expect("base srcloc must be set at this point")
    }

    fn clear(&mut self) {
        self.base_srcloc = None;
        self.user_named_funcs.clear();
    }
}

/// Function fields needed when compiling a function.
///
/// Additionally, these fields can be the same for two functions that would be compiled the same
/// way, and finalized by applying `FunctionParameters` onto their `CompiledCodeStencil`.
#[derive(Clone, PartialEq, Hash)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct FunctionStencil {
    /// A version marker used to ensure that serialized clif ir is never deserialized with a
    /// different version of Cranelift.
    // Note: This must be the first field to ensure that Serde will deserialize it before
    // attempting to deserialize other fields that are potentially changed between versions.
    pub version_marker: VersionMarker,

    /// Signature of this function.
    pub signature: Signature,

    /// Sized stack slots allocated in this function.
    pub sized_stack_slots: StackSlots,

    /// Dynamic stack slots allocated in this function.
    pub dynamic_stack_slots: DynamicStackSlots,

    /// Global values referenced.
    pub global_values: PrimaryMap<ir::GlobalValue, ir::GlobalValueData>,

    /// Heaps referenced.
    pub heaps: PrimaryMap<ir::Heap, ir::HeapData>,

    /// Tables referenced.
    pub tables: PrimaryMap<ir::Table, ir::TableData>,

    /// Jump tables used in this function.
    pub jump_tables: JumpTables,

    /// Data flow graph containing the primary definition of all instructions, blocks and values.
    pub dfg: DataFlowGraph,

    /// Layout of blocks and instructions in the function body.
    pub layout: Layout,

    /// Source locations.
    ///
    /// Track the original source location for each instruction. The source locations are not
    /// interpreted by Cranelift, only preserved.
    srclocs: SourceLocs,

    /// An optional global value which represents an expression evaluating to
    /// the stack limit for this function. This `GlobalValue` will be
    /// interpreted in the prologue, if necessary, to insert a stack check to
    /// ensure that a trap happens if the stack pointer goes below the
    /// threshold specified here.
    pub stack_limit: Option<ir::GlobalValue>,
}

impl FunctionStencil {
    fn clear(&mut self) {
        self.signature.clear(CallConv::Fast);
        self.sized_stack_slots.clear();
        self.dynamic_stack_slots.clear();
        self.global_values.clear();
        self.heaps.clear();
        self.tables.clear();
        self.jump_tables.clear();
        self.dfg.clear();
        self.layout.clear();
        self.srclocs.clear();
        self.stack_limit = None;
    }

    /// Creates a jump table in the function, to be used by `br_table` instructions.
    pub fn create_jump_table(&mut self, data: JumpTableData) -> JumpTable {
        self.jump_tables.push(data)
    }

    /// Creates a sized stack slot in the function, to be used by `stack_load`, `stack_store`
    /// and `stack_addr` instructions.
    pub fn create_sized_stack_slot(&mut self, data: StackSlotData) -> StackSlot {
        self.sized_stack_slots.push(data)
    }

    /// Creates a dynamic stack slot in the function, to be used by `dynamic_stack_load`,
    /// `dynamic_stack_store` and `dynamic_stack_addr` instructions.
    pub fn create_dynamic_stack_slot(&mut self, data: DynamicStackSlotData) -> DynamicStackSlot {
        self.dynamic_stack_slots.push(data)
    }

    /// Adds a signature which can later be used to declare an external function import.
    pub fn import_signature(&mut self, signature: Signature) -> SigRef {
        self.dfg.signatures.push(signature)
    }

    /// Declares a global value accessible to the function.
    pub fn create_global_value(&mut self, data: GlobalValueData) -> GlobalValue {
        self.global_values.push(data)
    }

    /// Find the global dyn_scale value associated with given DynamicType
    pub fn get_dyn_scale(&self, ty: DynamicType) -> GlobalValue {
        self.dfg.dynamic_types.get(ty).unwrap().dynamic_scale
    }

    /// Find the global dyn_scale for the given stack slot.
    pub fn get_dynamic_slot_scale(&self, dss: DynamicStackSlot) -> GlobalValue {
        let dyn_ty = self.dynamic_stack_slots.get(dss).unwrap().dyn_ty;
        self.get_dyn_scale(dyn_ty)
    }

    /// Get a concrete `Type` from a user defined `DynamicType`.
    pub fn get_concrete_dynamic_ty(&self, ty: DynamicType) -> Option<Type> {
        self.dfg
            .dynamic_types
            .get(ty)
            .unwrap_or_else(|| panic!("Undeclared dynamic vector type: {}", ty))
            .concrete()
    }

    /// Declares a heap accessible to the function.
    pub fn create_heap(&mut self, data: HeapData) -> Heap {
        self.heaps.push(data)
    }

    /// Declares a table accessible to the function.
    pub fn create_table(&mut self, data: TableData) -> Table {
        self.tables.push(data)
    }

    /// Find a presumed unique special-purpose function parameter value.
    ///
    /// Returns the value of the last `purpose` parameter, or `None` if no such parameter exists.
    pub fn special_param(&self, purpose: ir::ArgumentPurpose) -> Option<ir::Value> {
        let entry = self.layout.entry_block().expect("Function is empty");
        self.signature
            .special_param_index(purpose)
            .map(|i| self.dfg.block_params(entry)[i])
    }

    /// Starts collection of debug information.
    pub fn collect_debug_info(&mut self) {
        self.dfg.collect_debug_info();
    }

    /// Changes the destination of a jump or branch instruction.
    /// Does nothing if called with a non-jump or non-branch instruction.
    ///
    /// Note that this method ignores multi-destination branches like `br_table`.
    pub fn change_branch_destination(&mut self, inst: Inst, new_dest: Block) {
        match self.dfg[inst].branch_destination_mut() {
            None => (),
            Some(inst_dest) => *inst_dest = new_dest,
        }
    }

    /// Rewrite the branch destination to `new_dest` if the destination matches `old_dest`.
    /// Does nothing if called with a non-jump or non-branch instruction.
    ///
    /// Unlike [change_branch_destination](Function::change_branch_destination), this method rewrite the destinations of
    /// multi-destination branches like `br_table`.
    pub fn rewrite_branch_destination(&mut self, inst: Inst, old_dest: Block, new_dest: Block) {
        match self.dfg.analyze_branch(inst) {
            BranchInfo::SingleDest(dest, ..) => {
                if dest == old_dest {
                    self.change_branch_destination(inst, new_dest);
                }
            }

            BranchInfo::Table(table, default_dest) => {
                self.jump_tables[table].iter_mut().for_each(|entry| {
                    if *entry == old_dest {
                        *entry = new_dest;
                    }
                });

                if default_dest == Some(old_dest) {
                    match &mut self.dfg[inst] {
                        InstructionData::BranchTable { destination, .. } => {
                            *destination = new_dest;
                        }
                        _ => panic!(
                            "Unexpected instruction {} having default destination",
                            self.dfg.display_inst(inst)
                        ),
                    }
                }
            }

            BranchInfo::NotABranch => {}
        }
    }

    /// Checks that the specified block can be encoded as a basic block.
    ///
    /// On error, returns the first invalid instruction and an error message.
    pub fn is_block_basic(&self, block: Block) -> Result<(), (Inst, &'static str)> {
        let dfg = &self.dfg;
        let inst_iter = self.layout.block_insts(block);

        // Ignore all instructions prior to the first branch.
        let mut inst_iter = inst_iter.skip_while(|&inst| !dfg[inst].opcode().is_branch());

        // A conditional branch is permitted in a basic block only when followed
        // by a terminal jump instruction.
        if let Some(_branch) = inst_iter.next() {
            if let Some(next) = inst_iter.next() {
                match dfg[next].opcode() {
                    Opcode::Jump => (),
                    _ => return Err((next, "post-branch instruction not jump")),
                }
            }
        }

        Ok(())
    }

    /// Returns true if the function is function that doesn't call any other functions. This is not
    /// to be confused with a "leaf function" in Windows terminology.
    pub fn is_leaf(&self) -> bool {
        // Conservative result: if there's at least one function signature referenced in this
        // function, assume it is not a leaf.
        self.dfg.signatures.is_empty()
    }

    /// Replace the `dst` instruction's data with the `src` instruction's data
    /// and then remove `src`.
    ///
    /// `src` and its result values should not be used at all, as any uses would
    /// be left dangling after calling this method.
    ///
    /// `src` and `dst` must have the same number of resulting values, and
    /// `src`'s i^th value must have the same type as `dst`'s i^th value.
    pub fn transplant_inst(&mut self, dst: Inst, src: Inst) {
        debug_assert_eq!(
            self.dfg.inst_results(dst).len(),
            self.dfg.inst_results(src).len()
        );
        debug_assert!(self
            .dfg
            .inst_results(dst)
            .iter()
            .zip(self.dfg.inst_results(src))
            .all(|(a, b)| self.dfg.value_type(*a) == self.dfg.value_type(*b)));

        self.dfg[dst] = self.dfg[src].clone();
        self.layout.remove_inst(src);
    }

    /// Size occupied by all stack slots associated with this function.
    ///
    /// Does not include any padding necessary due to offsets
    pub fn fixed_stack_size(&self) -> u32 {
        self.sized_stack_slots.values().map(|ss| ss.size).sum()
    }

    /// Returns the list of relative source locations for this function.
    pub fn rel_srclocs(&self) -> &SecondaryMap<Inst, RelSourceLoc> {
        &self.srclocs
    }
}

/// Functions can be cloned, but it is not a very fast operation.
/// The clone will have all the same entity numbers as the original.
#[derive(Clone)]
#[cfg_attr(feature = "enable-serde", derive(Serialize, Deserialize))]
pub struct Function {
    /// All the fields required for compiling a function, independently of details irrelevant to
    /// compilation and that are stored in the `FunctionParameters` `params` field instead.
    pub stencil: FunctionStencil,

    /// All the parameters that can be applied onto the function stencil, that is, that don't
    /// matter when caching compilation artifacts.
    pub params: FunctionParameters,
}

impl core::ops::Deref for Function {
    type Target = FunctionStencil;

    fn deref(&self) -> &Self::Target {
        &self.stencil
    }
}

impl core::ops::DerefMut for Function {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.stencil
    }
}

impl Function {
    /// Create a function with the given name and signature.
    pub fn with_name_signature(name: ExternalName, sig: Signature) -> Self {
        Self {
            stencil: FunctionStencil {
                version_marker: VersionMarker,
                signature: sig,
                sized_stack_slots: StackSlots::new(),
                dynamic_stack_slots: DynamicStackSlots::new(),
                global_values: PrimaryMap::new(),
                heaps: PrimaryMap::new(),
                tables: PrimaryMap::new(),
                jump_tables: PrimaryMap::new(),
                dfg: DataFlowGraph::new(),
                layout: Layout::new(),
                srclocs: SecondaryMap::new(),
                stack_limit: None,
            },
            params: FunctionParameters::new(name),
        }
    }

    /// Clear all data structures in this function.
    pub fn clear(&mut self) {
        self.stencil.clear();
        self.params.clear();
    }

    /// Create a new empty, anonymous function with a Fast calling convention.
    pub fn new() -> Self {
        Self::with_name_signature(Default::default(), Signature::new(CallConv::Fast))
    }

    /// Return an object that can display this function with correct ISA-specific annotations.
    pub fn display(&self) -> DisplayFunction<'_> {
        DisplayFunction(self, Default::default())
    }

    /// Return an object that can display this function with correct ISA-specific annotations.
    pub fn display_with<'a>(
        &'a self,
        annotations: DisplayFunctionAnnotations<'a>,
    ) -> DisplayFunction<'a> {
        DisplayFunction(self, annotations)
    }

    /// Sets an absolute source location for the given instruction.
    pub fn set_srcloc(&mut self, inst: Inst, srcloc: SourceLoc) {
        if self.params.base_srcloc.is_none() {
            self.params.base_srcloc = Some(srcloc);
        }
        self.stencil.srclocs[inst] =
            RelSourceLoc::from_base_offset(self.params.base_srcloc.unwrap(), srcloc);
    }

    /// Returns an absolute source location for the given instruction.
    pub fn srcloc(&self, inst: Inst) -> SourceLoc {
        self.stencil.srclocs[inst].expand(self.params.base_srcloc())
    }

    /// Sets a relative source location for the given instruction.
    pub(crate) fn set_rel_srcloc(&mut self, inst: Inst, srcloc: RelSourceLoc) {
        self.stencil.srclocs[inst] = srcloc;
    }

    /// Declare a user-defined external function import, to be referenced in `ExtFuncData::User` later.
    pub fn declare_imported_user_function(
        &mut self,
        name: UserExternalName,
    ) -> UserExternalNameRef {
        self.params.user_named_funcs.push(name)
    }

    /// Declare an external function import.
    pub fn import_function(&mut self, data: ExtFuncData) -> FuncRef {
        self.stencil.dfg.ext_funcs.push(data)
    }
}

/// Additional annotations for function display.
#[derive(Default)]
pub struct DisplayFunctionAnnotations<'a> {
    /// Enable value labels annotations.
    pub value_ranges: Option<&'a ValueLabelsRanges>,
}

/// Wrapper type capable of displaying a `Function` with correct ISA annotations.
pub struct DisplayFunction<'a>(&'a Function, DisplayFunctionAnnotations<'a>);

impl<'a> fmt::Display for DisplayFunction<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write_function(fmt, self.0)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write_function(fmt, self)
    }
}

impl fmt::Debug for Function {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write_function(fmt, self)
    }
}
