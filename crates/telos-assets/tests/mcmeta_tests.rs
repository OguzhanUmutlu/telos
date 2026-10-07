//! Unit tests for `.mcmeta` parsing and animated texture baking.

use telos_assets::{AnimationDef, AnimationFrameDef, RgbaImage, TextureArrayBuilder};

#[test]
fn test_mcmeta_parse_standard() {
    let json = r#"{
        "animation": {
            "frametime": 2,
            "interpolate": true,
            "frames": [0, 1, 2, 3, 2, 1]
        }
    }"#;

    let anim = AnimationDef::from_json_str(json).expect("valid mcmeta JSON");
    assert_eq!(anim.frametime, 2);
    assert!(anim.interpolate);
    assert_eq!(anim.resolve_frames(4), vec![0, 1, 2, 3, 2, 1]);
}

#[test]
fn test_mcmeta_parse_empty_animation_object() {
    let json = r#"{ "animation": {} }"#;
    let anim = AnimationDef::from_json_str(json).expect("valid empty animation");
    assert_eq!(anim.frametime, 1);
    assert!(!anim.interpolate);
    assert_eq!(anim.resolve_frames(3), vec![0, 1, 2]);
}

#[test]
fn test_mcmeta_parse_detailed_frames() {
    let json = r#"{
        "animation": {
            "frametime": 4,
            "frames": [
                { "index": 0, "time": 10 },
                { "index": 1, "time": 4 }
            ]
        }
    }"#;

    let anim = AnimationDef::from_json_str(json).expect("valid detailed frames");
    assert_eq!(anim.frametime, 4);
    assert_eq!(anim.resolve_frames(2), vec![0, 1]);
}

#[test]
fn test_builder_insert_animated_with_meta() {
    let mut builder = TextureArrayBuilder::new(16);

    let frame0 = RgbaImage::new(16, 16);
    let frame1 = RgbaImage::new(16, 16);
    let frame2 = RgbaImage::new(16, 16);
    let physical_frames = vec![frame0, frame1, frame2];

    let anim = AnimationDef {
        frametime: 3,
        interpolate: false,
        frames: Some(vec![
            AnimationFrameDef::Index(0),
            AnimationFrameDef::Index(1),
            AnimationFrameDef::Index(2),
            AnimationFrameDef::Index(1),
        ]),
    };

    let info = builder.insert_animated_with_meta("custom_lava", &physical_frames, Some(&anim));
    assert_eq!(info.base_layer, 0);
    assert_eq!(info.frame_count, 4);
    assert_eq!(info.frame_time, 3);
    assert_eq!(builder.layer_count(), 4);

    let baked = builder.bake();
    assert_eq!(baked.layer_count, 4);
}
