#![no_main]

use cranelift_codegen::{incremental_cache as icache, isa, settings, Context};
use libfuzzer_sys::fuzz_target;

use cranelift_fuzzgen::*;
use target_lexicon::Triple;

fuzz_target!(|func: SingleFunction| {
    let func = func.0;

    let flags = settings::Flags::new(settings::builder());

    let isa_builder = isa::lookup(Triple::host())
        .map_err(|err| match err {
            isa::LookupError::SupportDisabled => {
                "support for architecture disabled at compile time"
            }
            isa::LookupError::Unsupported => "unsupported architecture",
        })
        .unwrap();

    let isa = isa_builder.finish(flags).unwrap();

    let (cache_key, _cache_key_hash) = icache::compute_cache_key(&func);

    let mut context = Context::for_function(func.clone());
    let prev_info = match context.compile(&*isa) {
        Ok(info) => info,
        Err(_) => return,
    };

    let serialized = icache::serialize_compiled(
        cache_key.clone(),
        &func,
        context.mach_compile_result.as_ref().unwrap(),
    )
    .expect("serialization failure");

    let mut prev_assembly = vec![0; prev_info.total_size as usize];
    unsafe {
        context.emit_to_memory(prev_assembly.as_mut_ptr());
    }

    let new_result = icache::try_finish_recompile(&cache_key, &func, &serialized)
        .expect("recompilation should always work for identity");
    let new_info = new_result.code_info();

    assert_eq!(new_info, prev_info, "CodeInfo:s don't match");

    context.mach_compile_result = Some(new_result);
    let mut new_assembly = vec![0; new_info.total_size as usize];
    unsafe {
        context.emit_to_memory(new_assembly.as_mut_ptr());
    }

    assert_eq!(prev_assembly, new_assembly, "assembly buffers don't match");
});
