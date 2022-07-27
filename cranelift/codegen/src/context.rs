//! Cranelift compilation context and main entry point.
//!
//! When compiling many small functions, it is important to avoid repeatedly allocating and
//! deallocating the data structures needed for compilation. The `Context` struct is used to hold
//! on to memory allocations between function compilations.
//!
//! The context does not hold a `TargetIsa` instance which has to be provided as an argument
//! instead. This is because an ISA instance is immutable and can be used by multiple compilation
//! contexts concurrently. Typically, you would have one context per compilation thread and only a
//! single ISA instance.

use crate::alias_analysis::AliasAnalysis;
use crate::binemit::CodeInfo;
use crate::dce::do_dce;
use crate::dominator_tree::DominatorTree;
use crate::flowgraph::ControlFlowGraph;
use crate::ir::Function;
use crate::isa::TargetIsa;
use crate::legalizer::simple_legalize;
use crate::licm::do_licm;
use crate::loop_analysis::LoopAnalysis;
use crate::machinst::MachCompileResult;
use crate::nan_canonicalization::do_nan_canonicalization;
use crate::remove_constant_phis::do_remove_constant_phis;
use crate::result::CodegenResult;
use crate::settings::{FlagsOrIsa, OptLevel};
use crate::simple_gvn::do_simple_gvn;
use crate::simple_preopt::do_preopt;
use crate::timing;
use crate::unreachable_code::eliminate_unreachable_code;
use crate::verifier::{verify_context, VerifierErrors, VerifierResult};
#[cfg(feature = "souper-harvest")]
use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "souper-harvest")]
use crate::souper_harvest::do_souper_harvest;

/// Persistent data structures and compilation pipeline.
pub struct Context {
    /// The function we're compiling.
    pub func: Function,

    /// The control flow graph of `func`.
    pub cfg: ControlFlowGraph,

    /// Dominator tree for `func`.
    pub domtree: DominatorTree,

    /// Loop analysis of `func`.
    pub loop_analysis: LoopAnalysis,

    /// Result of MachBackend compilation, if computed.
    pub mach_compile_result: Option<MachCompileResult>,

    /// Flag: do we want a disassembly with the MachCompileResult?
    pub want_disasm: bool,

    /// TODO
    pub stats: IncrementalCacheStats,
}

/// TODO
#[derive(Default)]
pub struct IncrementalCacheStats {
    num_lookups: usize,
    num_hits: usize,
    num_cached: usize,
}

impl IncrementalCacheStats {
    /// TODO
    pub fn fuse(&mut self, other: &IncrementalCacheStats) {
        self.num_lookups += other.num_lookups;
        self.num_hits += other.num_hits;
        self.num_cached += other.num_cached;
    }

    /// TODO
    pub fn print(&mut self) {
        eprintln!(
            //log::debug!(
            "Incremental compilation cache stats: {}/{} = {}% (hits/lookup)\ncached: {}",
            self.num_hits,
            self.num_lookups,
            (self.num_hits as f32) / (self.num_lookups as f32) * 100.0,
            self.num_cached
        );
        self.num_hits = 0;
        self.num_lookups = 0;
        self.num_cached = 0;
    }
}

impl Context {
    /// Allocate a new compilation context.
    ///
    /// The returned instance should be reused for compiling multiple functions in order to avoid
    /// needless allocator thrashing.
    pub fn new() -> Self {
        Self::for_function(Function::new())
    }

    /// Allocate a new compilation context with an existing Function.
    ///
    /// The returned instance should be reused for compiling multiple functions in order to avoid
    /// needless allocator thrashing.
    pub fn for_function(func: Function) -> Self {
        Self {
            func,
            cfg: ControlFlowGraph::new(),
            domtree: DominatorTree::new(),
            loop_analysis: LoopAnalysis::new(),
            mach_compile_result: None,
            want_disasm: false,
            stats: Default::default(),
        }
    }

    /// Clear all data structures in this context.
    pub fn clear(&mut self) {
        self.func.clear();
        self.cfg.clear();
        self.domtree.clear();
        self.loop_analysis.clear();
        self.mach_compile_result = None;
        self.want_disasm = false;
    }

    /// Set the flag to request a disassembly when compiling with a
    /// `MachBackend` backend.
    pub fn set_disasm(&mut self, val: bool) {
        self.want_disasm = val;
    }

    /// Compile the function, and emit machine code into a `Vec<u8>`.
    ///
    /// Run the function through all the passes necessary to generate code for the target ISA
    /// represented by `isa`, as well as the final step of emitting machine code into a
    /// `Vec<u8>`. The machine code is not relocated. Instead, any relocations can be obtained
    /// from `mach_compile_result`.
    ///
    /// This function calls `compile` and `emit_to_memory`, taking care to resize `mem` as
    /// needed, so it provides a safe interface.
    ///
    /// Returns information about the function's code and read-only data.
    pub fn compile_and_emit(
        &mut self,
        isa: &dyn TargetIsa,
        mem: &mut Vec<u8>,
    ) -> CodegenResult<()> {
        let info = self.compile(isa)?;
        let old_len = mem.len();
        mem.resize(old_len + info.total_size as usize, 0);
        let new_info = unsafe { self.emit_to_memory(mem.as_mut_ptr().add(old_len)) };
        debug_assert_eq!(new_info, info);
        Ok(())
    }

    /// Compile the function.
    ///
    /// Run the function through all the passes necessary to generate code for the target ISA
    /// represented by `isa`. This does not include the final step of emitting machine code into a
    /// code sink.
    ///
    /// Returns information about the function's code and read-only data.
    pub fn compile(&mut self, isa: &dyn TargetIsa) -> CodegenResult<CodeInfo> {
        let _tt = timing::compile();

        self.verify_if(isa)?;

        let opt_level = isa.flags().opt_level();
        log::trace!(
            "Compiling (opt level {:?}):\n{}",
            opt_level,
            self.func.display()
        );

        self.compute_cfg();
        if opt_level != OptLevel::None {
            self.preopt(isa)?;
        }
        if isa.flags().enable_nan_canonicalization() {
            self.canonicalize_nans(isa)?;
        }

        self.legalize(isa)?;
        if opt_level != OptLevel::None {
            self.compute_domtree();
            self.compute_loop_analysis();
            self.licm(isa)?;
            self.simple_gvn(isa)?;
        }

        self.compute_domtree();
        self.eliminate_unreachable_code(isa)?;
        if opt_level != OptLevel::None {
            self.dce(isa)?;
        }

        self.remove_constant_phis(isa)?;

        if opt_level != OptLevel::None && isa.flags().enable_alias_analysis() {
            self.replace_redundant_loads()?;
            self.simple_gvn(isa)?;
        }

        let result = isa.compile_function(&self.func, self.want_disasm)?;

        let info = result.code_info();
        self.mach_compile_result = Some(result);
        Ok(info)
    }

    /// Compile the function, as in `compile`, but tries to reuse compiled artifacts of former
    /// compilations.
    ///
    /// Requires the Cranelift dynamic flag `enable_incremental_compilation_cache` to be enabled.
    #[cfg(feature = "incremental-cache")]
    pub fn compile_with_cache(
        &mut self,
        isa: &dyn TargetIsa,
        cache_store: &mut dyn crate::incremental_cache::CacheStore,
    ) -> CodegenResult<CodeInfo> {
        if !isa.flags().enable_incremental_compilation_cache() {
            // If the dynamic flag isn't enabled, compile without any caching involved.
            return self.compile(isa);
        }

        let (cache_key_hash, cache_key) = {
            let _tt = timing::try_incremental_cache();

            self.stats.num_lookups += 1;

            let (cache_key, cache_key_hash) = crate::incremental_cache::get_cache_key(&self.func);

            if let Some(blob) = cache_store.get(cache_key_hash) {
                if let Ok(mach_compile_result) =
                    crate::incremental_cache::try_finish_recompile(&cache_key, &self.func, &blob)
                {
                    let info = mach_compile_result.code_info();
                    self.mach_compile_result = Some(mach_compile_result);
                    self.stats.num_hits += 1;
                    return Ok(info);
                }
            }

            (cache_key_hash, cache_key)
        };

        let info = self.compile(isa)?;

        let result = self
            .mach_compile_result
            .as_ref()
            .expect("if compilation succeeds, then mach_compile_result is set");

        let _tt = timing::store_incremental_cache();
        if let Ok(blob) =
            crate::incremental_cache::serialize_compiled(cache_key, &self.func, result)
        {
            if cache_store.insert(cache_key_hash, blob) {
                self.stats.num_cached += 1;
            }
        }

        Ok(info)
    }

    /// Emit machine code directly into raw memory.
    ///
    /// Write all of the function's machine code to the memory at `mem`. The size of the machine
    /// code is returned by `compile` above.
    ///
    /// The machine code is not relocated.
    /// Instead, any relocations can be obtained from `mach_compile_result`.
    ///
    /// # Safety
    ///
    /// This function is unsafe since it does not perform bounds checking on the memory buffer,
    /// and it can't guarantee that the `mem` pointer is valid.
    ///
    /// Returns information about the emitted code and data.
    #[deny(unsafe_op_in_unsafe_fn)]
    pub unsafe fn emit_to_memory(&self, mem: *mut u8) -> CodeInfo {
        let _tt = timing::binemit();
        let result = self
            .mach_compile_result
            .as_ref()
            .expect("only using mach backend now");
        let info = result.code_info();

        let mem = unsafe { std::slice::from_raw_parts_mut(mem, info.total_size as usize) };
        mem.copy_from_slice(result.buffer.data());

        info
    }

    /// If available, return information about the code layout in the
    /// final machine code: the offsets (in bytes) of each basic-block
    /// start, and all basic-block edges.
    pub fn get_code_bb_layout(&self) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        if let Some(result) = self.mach_compile_result.as_ref() {
            Some((
                result.bb_starts.iter().map(|&off| off as usize).collect(),
                result
                    .bb_edges
                    .iter()
                    .map(|&(from, to)| (from as usize, to as usize))
                    .collect(),
            ))
        } else {
            None
        }
    }

    /// Creates unwind information for the function.
    ///
    /// Returns `None` if the function has no unwind information.
    #[cfg(feature = "unwind")]
    pub fn create_unwind_info(
        &self,
        isa: &dyn TargetIsa,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        let unwind_info_kind = isa.unwind_info_kind();
        let result = self.mach_compile_result.as_ref().unwrap();
        isa.emit_unwind_info(result, unwind_info_kind)
    }

    /// Run the verifier on the function.
    ///
    /// Also check that the dominator tree and control flow graph are consistent with the function.
    pub fn verify<'a, FOI: Into<FlagsOrIsa<'a>>>(&self, fisa: FOI) -> VerifierResult<()> {
        let mut errors = VerifierErrors::default();
        let _ = verify_context(&self.func, &self.cfg, &self.domtree, fisa, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Run the verifier only if the `enable_verifier` setting is true.
    pub fn verify_if<'a, FOI: Into<FlagsOrIsa<'a>>>(&self, fisa: FOI) -> CodegenResult<()> {
        let fisa = fisa.into();
        if fisa.flags.enable_verifier() {
            self.verify(fisa)?;
        }
        Ok(())
    }

    /// Perform dead-code elimination on the function.
    pub fn dce<'a, FOI: Into<FlagsOrIsa<'a>>>(&mut self, fisa: FOI) -> CodegenResult<()> {
        do_dce(&mut self.func, &mut self.domtree);
        self.verify_if(fisa)?;
        Ok(())
    }

    /// Perform constant-phi removal on the function.
    pub fn remove_constant_phis<'a, FOI: Into<FlagsOrIsa<'a>>>(
        &mut self,
        fisa: FOI,
    ) -> CodegenResult<()> {
        do_remove_constant_phis(&mut self.func, &mut self.domtree);
        self.verify_if(fisa)?;
        Ok(())
    }

    /// Perform pre-legalization rewrites on the function.
    pub fn preopt(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        do_preopt(&mut self.func, &mut self.cfg, isa);
        self.verify_if(isa)?;
        Ok(())
    }

    /// Perform NaN canonicalizing rewrites on the function.
    pub fn canonicalize_nans(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        do_nan_canonicalization(&mut self.func);
        self.verify_if(isa)
    }

    /// Run the legalizer for `isa` on the function.
    pub fn legalize(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        // Legalization invalidates the domtree and loop_analysis by mutating the CFG.
        // TODO: Avoid doing this when legalization doesn't actually mutate the CFG.
        self.domtree.clear();
        self.loop_analysis.clear();

        // Run some specific legalizations only.
        simple_legalize(&mut self.func, &mut self.cfg, isa);
        self.verify_if(isa)
    }

    /// Compute the control flow graph.
    pub fn compute_cfg(&mut self) {
        self.cfg.compute(&self.func)
    }

    /// Compute dominator tree.
    pub fn compute_domtree(&mut self) {
        self.domtree.compute(&self.func, &self.cfg)
    }

    /// Compute the loop analysis.
    pub fn compute_loop_analysis(&mut self) {
        self.loop_analysis
            .compute(&self.func, &self.cfg, &self.domtree)
    }

    /// Compute the control flow graph and dominator tree.
    pub fn flowgraph(&mut self) {
        self.compute_cfg();
        self.compute_domtree()
    }

    /// Perform simple GVN on the function.
    pub fn simple_gvn<'a, FOI: Into<FlagsOrIsa<'a>>>(&mut self, fisa: FOI) -> CodegenResult<()> {
        do_simple_gvn(&mut self.func, &mut self.domtree);
        self.verify_if(fisa)
    }

    /// Perform LICM on the function.
    pub fn licm(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        do_licm(
            &mut self.func,
            &mut self.cfg,
            &mut self.domtree,
            &mut self.loop_analysis,
        );
        self.verify_if(isa)
    }

    /// Perform unreachable code elimination.
    pub fn eliminate_unreachable_code<'a, FOI>(&mut self, fisa: FOI) -> CodegenResult<()>
    where
        FOI: Into<FlagsOrIsa<'a>>,
    {
        eliminate_unreachable_code(&mut self.func, &mut self.cfg, &self.domtree);
        self.verify_if(fisa)
    }

    /// Replace all redundant loads with the known values in
    /// memory. These are loads whose values were already loaded by
    /// other loads earlier, as well as loads whose values were stored
    /// by a store instruction to the same instruction (so-called
    /// "store-to-load forwarding").
    pub fn replace_redundant_loads(&mut self) -> CodegenResult<()> {
        let mut analysis = AliasAnalysis::new(&mut self.func, &self.domtree);
        analysis.compute_and_update_aliases();
        Ok(())
    }

    /// Harvest candidate left-hand sides for superoptimization with Souper.
    #[cfg(feature = "souper-harvest")]
    pub fn souper_harvest(
        &mut self,
        out: &mut std::sync::mpsc::Sender<String>,
    ) -> CodegenResult<()> {
        do_souper_harvest(&self.func, out);
        Ok(())
    }
}
