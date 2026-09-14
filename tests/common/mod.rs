#![allow(dead_code)]

pub fn segmenter() -> nagisa_rs::JaSegmenter {
    if let Some(dir) = std::env::var_os("NAGISA_RS_MODEL_DIR") {
        return nagisa_rs::JaSegmenter::from_nagisa_dir(dir).expect("load original model");
    }
    #[cfg(feature = "bundled-model")]
    return nagisa_rs::JaSegmenter::new().expect("load bundled model");
    #[cfg(not(feature = "bundled-model"))]
    panic!("enable bundled-model or set NAGISA_RS_MODEL_DIR");
}

pub fn tagger() -> nagisa_rs::Tagger {
    if let Some(dir) = std::env::var_os("NAGISA_RS_MODEL_DIR") {
        return nagisa_rs::Tagger::from_nagisa_dir(dir).expect("load original model");
    }
    #[cfg(feature = "bundled-model")]
    return nagisa_rs::Tagger::new().expect("load bundled model");
    #[cfg(not(feature = "bundled-model"))]
    panic!("enable bundled-model or set NAGISA_RS_MODEL_DIR");
}
