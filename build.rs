//! Turns the "do we have the data files?" question into a compile-time cfg so
//! the tests that need them can be `#[ignore]`d *with a reason* instead of
//! silently returning `ok`.
//!
//! Model parity runs with the default bundled model or an explicit original
//! model directory. Only original-file comparisons and Unicode audits need
//! additional local data. This script performs no downloads or model generation.

fn main() {
    for var in [
        "RAGISA_MODEL_DIR",
        "RAGISA_UNICODE_REF",
        "RAGISA_UNICODE_SWEEPS",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    for cfg in ["have_nagisa_dir", "have_unicode_ref", "have_unicode_sweeps"] {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }
    if std::env::var_os("RAGISA_MODEL_DIR").is_some() {
        println!("cargo::rustc-cfg=have_nagisa_dir");
    }
    if std::env::var_os("RAGISA_UNICODE_REF").is_some() {
        println!("cargo::rustc-cfg=have_unicode_ref");
    }
    if std::env::var_os("RAGISA_UNICODE_SWEEPS").is_some() {
        println!("cargo::rustc-cfg=have_unicode_sweeps");
    }
}
