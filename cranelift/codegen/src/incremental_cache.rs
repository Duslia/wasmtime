//! This module provides a set of primitives that allow implementing an incremental cache on top of
//! Cranelift, making it possible to reuse previous compiled artifacts for functions that have been
//! compiled previously.
//!
//! This set of operation is experimental and can be enabled using the Cargo feature
//! `incremental-cache`.
//!
//! This can bring speedups in different cases: change-code-and-immediately-recompile iterations
//! get faster, modules sharing lots of code can reuse each other's artifacts, etc.
//!
//! The three main primitives are the following:
//! - `compute_cache_key` is used to compute the cache key associated to a `Function`. This is
//! basically the content of the function, modulo a few things the caching system is resilient to.
//! - `serialize_compiled` is used to serialize the result of a compilation, so it can be reused
//! later on by...
//! - `try_finish_recompile`, which reads binary blobs serialized with `serialize_compiled`,
//! re-creating the compilation artifact from those.
//!
//! The `CacheStore` trait and `Context::compile_with_cache` method are provided as
//! high-level, easy-to-use facilities to make use of that cache, and show an example of how to use
//! the above three primitives to form a full incremental caching system.

use crate::alloc::string::String;
use crate::alloc::vec::Vec;
use crate::binemit::CodeOffset;
use crate::ir::function::FunctionStencil;
use crate::ir::{DynamicStackSlot, Function, StackSlot};
use crate::machinst::{MachBufferFinalized, MachCompileResult, MachCompileResultBase, Stencil};
use crate::ValueLabelsRanges;
use crate::{binemit::CodeInfo, isa::TargetIsa, timing, CodegenResult};
use crate::{trace, Context};
use alloc::borrow::{Cow, ToOwned};
use alloc::string::ToString as _;
use cranelift_entity::PrimaryMap;

impl Context {
    /// Compile the function, as in `compile`, but tries to reuse compiled artifacts from former
    /// compilations using the provided cache store.
    pub fn compile_with_cache(
        &mut self,
        isa: &dyn TargetIsa,
        cache_store: &mut dyn CacheKvStore,
    ) -> CodegenResult<(CodeInfo, bool)> {
        let (cache_key_hash, cache_key) = {
            let _tt = timing::try_incremental_cache();

            let (cache_key, cache_key_hash) = compute_cache_key(isa, &self.func);

            if let Some(blob) = cache_store.get(&cache_key_hash.0) {
                if let Ok(compiled_code) = try_finish_recompile(&cache_key, &self.func, &blob) {
                    let info = compiled_code.code_info();

                    if isa.flags().enable_incremental_compilation_cache_checks() {
                        let actual_info = self.compile(isa)?;
                        let actual_result = self
                            .mach_compile_result
                            .as_ref()
                            .expect("if compilation succeeds, then mach_compile_result is set");
                        assert_eq!(*actual_result, compiled_code);
                        assert_eq!(actual_info, info);
                    }

                    self.mach_compile_result = Some(compiled_code);
                    return Ok((info, true));
                }
            }

            (cache_key_hash, cache_key)
        };

        let stencil = self.compile_stencil(isa)?;

        let _tt = timing::store_incremental_cache();
        if let Ok(blob) = serialize_compiled(cache_key, &stencil) {
            cache_store.insert(&cache_key_hash.0, blob);
        }

        let result = self
            .mach_compile_result
            .insert(stencil.apply_params(&self.func.params));
        let info = result.code_info();

        Ok((info, false))
    }
}

/// Backing storage for an incremental compilation cache, when enabled.
pub trait CacheKvStore {
    /// Given a cache key hash, retrieves the associated opaque serialized data.
    fn get(&self, key: &[u8]) -> Option<Cow<[u8]>>;

    /// Given a new cache key and a serialized blob obtained from `serialize_compiled`, stores it
    /// in the cache store.
    fn insert(&mut self, key: &[u8], val: Vec<u8>);
}

/// Hashed `CachedKey`, to use as an identifier when looking up whether a function has already been
/// compiled or not.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub struct CacheKeyHash([u8; 8]);

impl std::fmt::Display for CacheKeyHash {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        i64::from_le_bytes(self.0).fmt(f)
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedFunc {
    cache_key: CacheKey,
    stencil: MachCompileResultBase<Stencil>,
}

/// Key for caching a single function's compilation.
///
/// If two functions get the same `CacheKey`, then we can reuse the compiled artifacts, modulo some
/// relocation fixups.
///
/// Everything in a `Function` that uniquely identifies a function must be included in this data
/// structure. For that matter, there's a method `check_from_func`.
#[derive(Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CacheKey {
    stencil: FunctionStencil,
    parameters: CompileParameters,
}

#[derive(Clone, PartialEq, Hash, serde::Serialize, serde::Deserialize)]
struct CompileParameters {
    isa: String,
    triple: String,
    flags: String,
    isa_flags: Vec<String>,
}

impl CompileParameters {
    fn from_isa(isa: &dyn TargetIsa) -> Self {
        Self {
            isa: isa.name().to_owned(),
            triple: isa.triple().to_string(),
            flags: isa.flags().to_string(),
            isa_flags: isa
                .isa_flags()
                .into_iter()
                .map(|v| v.value_string())
                .collect(),
        }
    }
}

impl CacheKey {
    /// Creates a new cache store key for a function.
    ///
    /// This is a bit expensive to compute, so it should be cached and reused as much as possible.
    fn new(isa: &dyn TargetIsa, f: &Function) -> Self {
        let mut stencil = f.stencil.clone();
        // Make sure the blocks and instructions are sequenced the same way as we might
        // have serialized them earlier. This is the symmetric of what's done in
        // `try_load`.
        stencil.layout.full_renumber();
        CacheKey {
            stencil,
            parameters: CompileParameters::from_isa(isa),
        }
    }
}

// --
// Our final data structure.

#[derive(Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct UserExternalName {
    namespace: u32,
    index: u32,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct CachedMachCompileResult {
    // These are fields from the `MachCompileResult`.
    // ---
    buffer: MachBufferFinalized<Stencil>,
    frame_size: u32,
    disasm: Option<String>,
    value_labels_ranges: ValueLabelsRanges,
    sized_stackslot_offsets: PrimaryMap<StackSlot, u32>,
    dynamic_stackslot_offsets: PrimaryMap<DynamicStackSlot, u32>,
    bb_starts: Vec<CodeOffset>,
    bb_edges: Vec<(CodeOffset, CodeOffset)>,
    // These are extra fields useful for patching purposes.
    // ---
    // /// Inverted mapping of user external name to `FuncRef`, constructed once to allow patching.
    //func_refs: HashMap<UserExternalName, FuncRef>,
}

/// Compute a cache key, and hash it on your behalf.
///
/// Since computing the `CacheKey` is a bit expensive, it should be done as least as possible.
#[inline(never)]
pub fn compute_cache_key(isa: &dyn TargetIsa, func: &Function) -> (CacheKey, CacheKeyHash) {
    let cache_key = CacheKey::new(isa, func);

    let hash = {
        use core::hash::{Hash as _, Hasher as _};
        let mut hasher = crate::fx::FxHasher::default();
        cache_key.hash(&mut hasher);
        hasher.finish()
    };

    (cache_key, CacheKeyHash(hash.to_le_bytes()))
}

/// Given a function that's been successfully compiled, serialize it to a blob that the caller may
/// store somewhere for future use by `try_finish_recompile`.
#[inline(never)]
pub fn serialize_compiled(
    cache_key: CacheKey,
    result: &MachCompileResultBase<Stencil>,
) -> Result<Vec<u8>, bincode::Error> {
    let cached = CachedFunc {
        cache_key,
        stencil: result.clone(),
    };
    bincode::serialize(&cached)
}

/// Given a function that's been precompiled and its entry in the caching storage, try to shortcut
/// compilation of the given function.
#[inline(never)]
pub fn try_finish_recompile(
    cache_key: &CacheKey,
    func: &Function,
    bytes: &[u8],
) -> Result<MachCompileResult, ()> {
    // try to deserialize, if not failure, return final recompiled code
    match bincode::deserialize::<CachedFunc>(bytes) {
        Ok(mut result) => {
            // Make sure the blocks and instructions are sequenced the same way as we might
            // have serialized them earlier. This is the symmetric of what's done in
            // `CacheKey`'s ctor.
            result.cache_key.stencil.layout.full_renumber();

            if *cache_key == result.cache_key {
                return Ok(result.stencil.apply_params(&func.params));
            }

            trace!("{} not read from cache: source mismatch", func.params.name);
        }

        Err(err) => {
            trace!("Couldn't deserialize cache entry: {err}");
        }
    }

    Err(())
}
