//! Turns the "do we have the data files?" question into a compile-time cfg so
//! the tests that need them can be `#[ignore]`d *with a reason* instead of
//! silently returning `ok`.
//!
//! nagisa's weights are not in this repository, so from a fresh clone the
//! differential test cannot run at all. Reporting that as a pass is how a
//! Unicode-table bug survived a green test run once already; reporting it as
//! `ignored, set NAGISA_RS_MODEL_DIR ...` does not.

fn main() {
    for var in [
        "NAGISA_RS_MODEL_DIR",
        "NAGISA_RS_UNICODE_REF",
        "NAGISA_RS_UNICODE_SWEEPS",
    ] {
        println!("cargo::rerun-if-env-changed={var}");
    }
    for cfg in ["have_nagisa_dir", "have_unicode_ref", "have_unicode_sweeps"] {
        println!("cargo::rustc-check-cfg=cfg({cfg})");
    }
    if std::env::var_os("NAGISA_RS_MODEL_DIR").is_some() {
        println!("cargo::rustc-cfg=have_nagisa_dir");
    }
    if std::env::var_os("NAGISA_RS_UNICODE_REF").is_some() {
        println!("cargo::rustc-cfg=have_unicode_ref");
    }
    if std::env::var_os("NAGISA_RS_UNICODE_SWEEPS").is_some() {
        println!("cargo::rustc-cfg=have_unicode_sweeps");
    }
}
