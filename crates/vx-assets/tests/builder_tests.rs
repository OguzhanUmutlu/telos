//! Unit tests for `TextureArrayBuilder` and mipmap generation.

use vx_assets::{MaterialTextureMap, RgbaImage, TextureArrayBuilder};

#[test]
fn test_image_mips_generation() {
    let mut img = RgbaImage::new(16, 16);
    // Fill with pattern
    for i in (0..img.data.len()).step_by(4) {
        img.data[i] = 200; // R
        img.data[i + 1] = 100; // G
        img.data[i + 2] = 50; // B
        img.data[i + 3] = 255; // A
    }

    let mips = img.generate_mips();
    // 16x16 -> 8x8 -> 4x4 -> 2x2 -> 1x1 = 5 levels
    assert_eq!(mips.len(), 5);
    assert_eq!(mips[0].width, 16);
    assert_eq!(mips[1].width, 8);
    assert_eq!(mips[2].width, 4);
    assert_eq!(mips[3].width, 2);
    assert_eq!(mips[4].width, 1);
}

#[test]
fn test_texture_array_baking() {
    let mut builder = TextureArrayBuilder::new(16);

    let red = RgbaImage::new(16, 16);
    let blue = RgbaImage::new(32, 32); // Different resolution, should be rescaled to 16

    let idx0 = builder.insert("red", red);
    let idx1 = builder.insert("blue", blue);
    let idx0_again = builder.insert("red", RgbaImage::new(16, 16));

    assert_eq!(idx0, 0);
    assert_eq!(idx1, 1);
    assert_eq!(idx0_again, 0);
    assert_eq!(builder.layer_count(), 2);

    let baked = builder.bake();
    assert_eq!(baked.resolution, 16);
    assert_eq!(baked.layer_count, 2);
    assert_eq!(baked.mip_levels, 5);
    assert!(!baked.pixel_data.is_empty());
    // 2 layers * 5 mips = 10 copy regions
    assert_eq!(baked.copy_regions.len(), 10);
}

#[test]
fn test_material_texture_map() {
    let mut map = MaterialTextureMap::new();
    map.set_uniform(1, 0); // Stone -> layer 0
    map.set_uniform(2, 1); // Dirt -> layer 1
    // Grass: top = 2, bottom = 1, sides = 3
    map.set(3, 2, 2); // PosY
    map.set(3, 3, 1); // NegY
    map.set(3, 0, 3); // PosX
    map.set(3, 1, 3); // NegX
    map.set(3, 4, 3); // PosZ
    map.set(3, 5, 3); // NegZ

    assert_eq!(map.get(1, 0), 0);
    assert_eq!(map.get(2, 4), 1);
    assert_eq!(map.get(3, 2), 2);
    assert_eq!(map.get(3, 3), 1);
    assert_eq!(map.get(3, 0), 3);

    let table = map.to_flat_table(3);
    assert_eq!(table.len(), 24); // (3 + 1) * 6
    assert_eq!(table[6], 0);
    assert_eq!(table[20], 2);
}
