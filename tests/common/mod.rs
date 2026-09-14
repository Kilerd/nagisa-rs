#![allow(dead_code)]

pub fn segmenter() -> ragisa::JaSegmenter {
    if let Some(dir) = std::env::var_os("RAGISA_MODEL_DIR") {
        return ragisa::JaSegmenter::from_nagisa_dir(dir).expect("load original model");
    }
    #[cfg(feature = "bundled-model")]
    return ragisa::JaSegmenter::new().expect("load bundled model");
    #[cfg(not(feature = "bundled-model"))]
    panic!("enable bundled-model or set RAGISA_MODEL_DIR");
}

pub fn tagger() -> ragisa::Tagger {
    if let Some(dir) = std::env::var_os("RAGISA_MODEL_DIR") {
        return ragisa::Tagger::from_nagisa_dir(dir).expect("load original model");
    }
    #[cfg(feature = "bundled-model")]
    return ragisa::Tagger::new().expect("load bundled model");
    #[cfg(not(feature = "bundled-model"))]
    panic!("enable bundled-model or set RAGISA_MODEL_DIR");
}
