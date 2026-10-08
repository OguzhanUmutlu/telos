//! Integration tests for enchantment rules and gameplay modifiers.

#![allow(clippy::float_cmp)]

use telos_sim::attributes::DamageType;
use telos_sim::enchantment::{
    CompactEnchantments, EnchantmentKind, calculate_arrow_damage, calculate_arrow_knockback,
    calculate_fire_aspect_seconds, calculate_knockback_bonus, calculate_melee_damage,
    calculate_mining_speed, calculate_total_epf, has_infinite_arrows, is_arrow_flaming,
    should_prevent_durability_loss,
};

#[test]
fn test_all_24_enchantment_kinds_roundtrip_and_properties() {
    for id in 1..=24u8 {
        let kind = EnchantmentKind::from_u8(id).expect("all 24 kinds must resolve");
        assert_eq!(kind.id(), id);
        assert!(!kind.name().is_empty());
        assert!(kind.max_level() >= 1 && kind.max_level() <= 5);
        assert!(kind.rarity_multiplier() >= 1);
    }
}

#[test]
fn test_mutual_exclusivity_matrix() {
    // Protections
    assert!(!EnchantmentKind::Protection.is_compatible_with(EnchantmentKind::FireProtection));
    assert!(!EnchantmentKind::Protection.is_compatible_with(EnchantmentKind::BlastProtection));
    assert!(!EnchantmentKind::Protection.is_compatible_with(EnchantmentKind::ProjectileProtection));
    assert!(!EnchantmentKind::FireProtection.is_compatible_with(EnchantmentKind::BlastProtection));

    // Damage
    assert!(!EnchantmentKind::Sharpness.is_compatible_with(EnchantmentKind::Smite));
    assert!(!EnchantmentKind::Sharpness.is_compatible_with(EnchantmentKind::BaneOfArthropods));
    assert!(!EnchantmentKind::Smite.is_compatible_with(EnchantmentKind::BaneOfArthropods));

    // Drops
    assert!(!EnchantmentKind::SilkTouch.is_compatible_with(EnchantmentKind::Fortune));

    // Bow infinity & mending
    assert!(!EnchantmentKind::Infinity.is_compatible_with(EnchantmentKind::Mending));

    // Independent enchantments
    assert!(EnchantmentKind::Sharpness.is_compatible_with(EnchantmentKind::Unbreaking));
    assert!(EnchantmentKind::Sharpness.is_compatible_with(EnchantmentKind::FireAspect));
    assert!(EnchantmentKind::Protection.is_compatible_with(EnchantmentKind::Unbreaking));
}

#[test]
fn test_epf_damage_reductions() {
    let mut helm = CompactEnchantments::new();
    helm.set_enchantment(EnchantmentKind::FireProtection, 4);

    let mut chest = CompactEnchantments::new();
    chest.set_enchantment(EnchantmentKind::Protection, 4);

    let mut legs = CompactEnchantments::new();
    legs.set_enchantment(EnchantmentKind::BlastProtection, 4);

    let mut boots = CompactEnchantments::new();
    boots.set_enchantment(EnchantmentKind::FeatherFalling, 4);

    // Fire damage: FireProtection (4 * 2 = 8) + Protection (4 * 1 = 4) = 12 EPF
    assert_eq!(
        calculate_total_epf(&[helm, chest, legs, boots], DamageType::Fire),
        12
    );

    // Fall damage: FeatherFalling (4 * 3 = 12) + Protection (4 * 1 = 4) = 16 EPF
    assert_eq!(
        calculate_total_epf(&[helm, chest, legs, boots], DamageType::Fall),
        16
    );

    // Attack damage: Protection (4 * 1 = 4)
    assert_eq!(
        calculate_total_epf(&[helm, chest, legs, boots], DamageType::Attack),
        4
    );
}

#[test]
fn test_gameplay_modifiers_comprehensive() {
    let mut sword = CompactEnchantments::new();
    sword.set_enchantment(EnchantmentKind::Sharpness, 5);
    sword.set_enchantment(EnchantmentKind::FireAspect, 2);
    sword.set_enchantment(EnchantmentKind::Knockback, 2);

    assert!((calculate_melee_damage(7.0, sword, false, false) - 10.0).abs() < 1e-4);
    assert_eq!(calculate_fire_aspect_seconds(sword), 8.0);
    assert_eq!(calculate_knockback_bonus(sword), 1.0);

    let mut tool = CompactEnchantments::new();
    tool.set_enchantment(EnchantmentKind::Efficiency, 5);
    tool.set_enchantment(EnchantmentKind::Unbreaking, 3);

    // Base 6.0 + 5^2 + 1 = 32.0
    assert_eq!(calculate_mining_speed(6.0, tool), 32.0);
    assert!(should_prevent_durability_loss(tool, 0.70));
    assert!(!should_prevent_durability_loss(tool, 0.80));

    let mut bow = CompactEnchantments::new();
    bow.set_enchantment(EnchantmentKind::Power, 5);
    bow.set_enchantment(EnchantmentKind::Punch, 2);
    bow.set_enchantment(EnchantmentKind::Flame, 1);
    bow.set_enchantment(EnchantmentKind::Infinity, 1);

    // 10.0 * (1.0 + 0.25 * 6 = 2.5) = 25.0
    assert_eq!(calculate_arrow_damage(10.0, bow), 25.0);
    assert_eq!(calculate_arrow_knockback(bow), 1.2);
    assert!(is_arrow_flaming(bow));
    assert!(has_infinite_arrows(bow));
}
