//! Potion types, brewing recipe matching, and potion consumption effects.

use crate::effect::{EffectInstance, StatusEffectKind};

/// Standard potion types supported in the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PotionType {
    /// Water bottle (base ingredient for awkward potion).
    Water = 0,
    /// Awkward potion (base for all primary effect potions).
    Awkward = 1,
    /// Mundane potion.
    Mundane = 2,
    /// Thick potion.
    Thick = 3,
    /// Potion of Swiftness (Speed I for 3:00 = 3600 ticks).
    Swiftness = 4,
    /// Potion of Swiftness II (Speed II for 1:30 = 1800 ticks).
    SwiftnessStrong = 5,
    /// Potion of Slowness (Slowness I for 1:30 = 1800 ticks).
    Slowness = 6,
    /// Potion of Strength (Strength I for 3:00 = 3600 ticks).
    Strength = 7,
    /// Potion of Strength II (Strength II for 1:30 = 1800 ticks).
    StrengthStrong = 8,
    /// Potion of Healing (Instant Health I).
    Healing = 9,
    /// Potion of Healing II (Instant Health II).
    HealingStrong = 10,
    /// Potion of Harming (Instant Damage I).
    Harming = 11,
    /// Potion of Harming II (Instant Damage II).
    HarmingStrong = 12,
    /// Potion of Poison (Poison I for 0:45 = 900 ticks).
    Poison = 13,
    /// Potion of Poison II (Poison II for 0:21 = 432 ticks).
    PoisonStrong = 14,
    /// Potion of Regeneration (Regeneration I for 0:45 = 900 ticks).
    Regeneration = 15,
    /// Potion of Regeneration II (Regeneration II for 0:22 = 440 ticks).
    RegenerationStrong = 16,
    /// Potion of Fire Resistance (Fire Resistance for 3:00 = 3600 ticks).
    FireResistance = 17,
    /// Potion of Invisibility (Invisibility for 3:00 = 3600 ticks).
    Invisibility = 18,
    /// Potion of Night Vision (Night Vision for 3:00 = 3600 ticks).
    NightVision = 19,
    /// Potion of Night Vision (Extended) (Night Vision for 8:00 = 9600 ticks).
    NightVisionLong = 20,
    /// Potion of Water Breathing (Water Breathing for 3:00 = 3600 ticks).
    WaterBreathing = 21,
    /// Potion of Water Breathing (Extended) (Water Breathing for 8:00 = 9600 ticks).
    WaterBreathingLong = 22,
    /// Potion of Leaping (Jump Boost I for 3:00 = 3600 ticks).
    Leaping = 23,
    /// Potion of Leaping II (Jump Boost II for 1:30 = 1800 ticks).
    LeapingStrong = 24,
    /// Potion of Weakness (Weakness for 1:30 = 1800 ticks).
    Weakness = 25,
    /// Potion of Weakness (Extended) (Weakness for 4:00 = 4800 ticks).
    WeaknessLong = 26,
}

impl PotionType {
    /// Numeric ID for serialization.
    #[must_use]
    pub const fn id(self) -> u8 {
        self as u8
    }

    /// Resolves numeric ID to `PotionType`.
    #[must_use]
    pub const fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::Water),
            1 => Some(Self::Awkward),
            2 => Some(Self::Mundane),
            3 => Some(Self::Thick),
            4 => Some(Self::Swiftness),
            5 => Some(Self::SwiftnessStrong),
            6 => Some(Self::Slowness),
            7 => Some(Self::Strength),
            8 => Some(Self::StrengthStrong),
            9 => Some(Self::Healing),
            10 => Some(Self::HealingStrong),
            11 => Some(Self::Harming),
            12 => Some(Self::HarmingStrong),
            13 => Some(Self::Poison),
            14 => Some(Self::PoisonStrong),
            15 => Some(Self::Regeneration),
            16 => Some(Self::RegenerationStrong),
            17 => Some(Self::FireResistance),
            18 => Some(Self::Invisibility),
            19 => Some(Self::NightVision),
            20 => Some(Self::NightVisionLong),
            21 => Some(Self::WaterBreathing),
            22 => Some(Self::WaterBreathingLong),
            23 => Some(Self::Leaping),
            24 => Some(Self::LeapingStrong),
            25 => Some(Self::Weakness),
            26 => Some(Self::WeaknessLong),
            _ => None,
        }
    }

    /// User-facing display name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Water => "Water Bottle",
            Self::Awkward => "Awkward Potion",
            Self::Mundane => "Mundane Potion",
            Self::Thick => "Thick Potion",
            Self::Swiftness => "Potion of Swiftness",
            Self::SwiftnessStrong => "Potion of Swiftness II",
            Self::Slowness => "Potion of Slowness",
            Self::Strength => "Potion of Strength",
            Self::StrengthStrong => "Potion of Strength II",
            Self::Healing => "Potion of Healing",
            Self::HealingStrong => "Potion of Healing II",
            Self::Harming => "Potion of Harming",
            Self::HarmingStrong => "Potion of Harming II",
            Self::Poison => "Potion of Poison",
            Self::PoisonStrong => "Potion of Poison II",
            Self::Regeneration => "Potion of Regeneration",
            Self::RegenerationStrong => "Potion of Regeneration II",
            Self::FireResistance => "Potion of Fire Resistance",
            Self::Invisibility => "Potion of Invisibility",
            Self::NightVision => "Potion of Night Vision",
            Self::NightVisionLong => "Potion of Night Vision (Extended)",
            Self::WaterBreathing => "Potion of Water Breathing",
            Self::WaterBreathingLong => "Potion of Water Breathing (Extended)",
            Self::Leaping => "Potion of Leaping",
            Self::LeapingStrong => "Potion of Leaping II",
            Self::Weakness => "Potion of Weakness",
            Self::WeaknessLong => "Potion of Weakness (Extended)",
        }
    }

    /// Status effect granted by drinking this potion.
    #[must_use]
    pub fn effect(self) -> Option<EffectInstance> {
        match self {
            Self::Water | Self::Awkward | Self::Mundane | Self::Thick => None,
            Self::Swiftness => Some(EffectInstance::new(StatusEffectKind::Speed, 3600, 0)),
            Self::SwiftnessStrong => Some(EffectInstance::new(StatusEffectKind::Speed, 1800, 1)),
            Self::Slowness => Some(EffectInstance::new(StatusEffectKind::Slowness, 1800, 0)),
            Self::Strength => Some(EffectInstance::new(StatusEffectKind::Strength, 3600, 0)),
            Self::StrengthStrong => Some(EffectInstance::new(StatusEffectKind::Strength, 1800, 1)),
            Self::Healing => Some(EffectInstance::new(StatusEffectKind::InstantHealth, 1, 0)),
            Self::HealingStrong => Some(EffectInstance::new(StatusEffectKind::InstantHealth, 1, 1)),
            Self::Harming => Some(EffectInstance::new(StatusEffectKind::InstantDamage, 1, 0)),
            Self::HarmingStrong => Some(EffectInstance::new(StatusEffectKind::InstantDamage, 1, 1)),
            Self::Poison => Some(EffectInstance::new(StatusEffectKind::Poison, 900, 0)),
            Self::PoisonStrong => Some(EffectInstance::new(StatusEffectKind::Poison, 432, 1)),
            Self::Regeneration => Some(EffectInstance::new(StatusEffectKind::Regeneration, 900, 0)),
            Self::RegenerationStrong => {
                Some(EffectInstance::new(StatusEffectKind::Regeneration, 440, 1))
            }
            Self::FireResistance => Some(EffectInstance::new(
                StatusEffectKind::FireResistance,
                3600,
                0,
            )),
            Self::Invisibility => {
                Some(EffectInstance::new(StatusEffectKind::Invisibility, 3600, 0))
            }
            Self::NightVision => Some(EffectInstance::new(StatusEffectKind::NightVision, 3600, 0)),
            Self::NightVisionLong => {
                Some(EffectInstance::new(StatusEffectKind::NightVision, 9600, 0))
            }
            Self::WaterBreathing => Some(EffectInstance::new(
                StatusEffectKind::WaterBreathing,
                3600,
                0,
            )),
            Self::WaterBreathingLong => Some(EffectInstance::new(
                StatusEffectKind::WaterBreathing,
                9600,
                0,
            )),
            Self::Leaping => Some(EffectInstance::new(StatusEffectKind::JumpBoost, 3600, 0)),
            Self::LeapingStrong => Some(EffectInstance::new(StatusEffectKind::JumpBoost, 1800, 1)),
            Self::Weakness => Some(EffectInstance::new(StatusEffectKind::Weakness, 1800, 0)),
            Self::WeaknessLong => Some(EffectInstance::new(StatusEffectKind::Weakness, 4800, 0)),
        }
    }
}

/// Resolves consumable status effects for an item ID.
#[must_use]
pub fn consume_item_effects(item_id: u32) -> Option<Vec<EffectInstance>> {
    match item_id {
        // Spider eye: Poison I for 4 seconds (80 ticks)
        44 => Some(vec![EffectInstance::new(StatusEffectKind::Poison, 80, 0)]),
        // Rotten Flesh: Slowness I for 5 seconds (100 ticks)
        48 => Some(vec![EffectInstance::new(
            StatusEffectKind::Slowness,
            100,
            0,
        )]),
        // Standard Potion bottle item (36) defaults to Speed I for 3:00 (3600 ticks)
        36 => Some(vec![EffectInstance::new(StatusEffectKind::Speed, 3600, 0)]),
        _ => None,
    }
}

/// A deterministic brewing recipe rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrewingRecipe {
    /// Starting base potion.
    pub base: PotionType,
    /// Ingredient item ID added at the top slot.
    pub ingredient: u32,
    /// Resulting potion.
    pub result: PotionType,
}

/// Registry storing all valid brewing transitions.
#[derive(Debug, Clone, PartialEq)]
pub struct BrewingRegistry {
    recipes: Vec<BrewingRecipe>,
}

impl Default for BrewingRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

impl BrewingRegistry {
    /// Creates a brewing registry populated with all standard recipes.
    ///
    /// Items used:
    /// - 40: Nether Wart
    /// - 41: Blaze Powder
    /// - 42: Sugar
    /// - 43: Glistering Melon
    /// - 44: Spider Eye
    /// - 45: Fermented Spider Eye
    /// - 46: Ghast Tear
    /// - 47: Magma Cream
    /// - 20: Logic Wire (Redstone analogue for extension/tiering)
    /// - 25: Logic Lamp (Glowstone analogue for Tier II)
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn standard() -> Self {
        let recipes = vec![
            // Base creation
            BrewingRecipe {
                base: PotionType::Water,
                ingredient: 40,
                result: PotionType::Awkward,
            },
            // Speed
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 42,
                result: PotionType::Swiftness,
            },
            BrewingRecipe {
                base: PotionType::Swiftness,
                ingredient: 25,
                result: PotionType::SwiftnessStrong,
            },
            BrewingRecipe {
                base: PotionType::Swiftness,
                ingredient: 45,
                result: PotionType::Slowness,
            },
            // Strength
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 41,
                result: PotionType::Strength,
            },
            BrewingRecipe {
                base: PotionType::Strength,
                ingredient: 25,
                result: PotionType::StrengthStrong,
            },
            // Healing & Harming
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 43,
                result: PotionType::Healing,
            },
            BrewingRecipe {
                base: PotionType::Healing,
                ingredient: 25,
                result: PotionType::HealingStrong,
            },
            BrewingRecipe {
                base: PotionType::Healing,
                ingredient: 45,
                result: PotionType::Harming,
            },
            BrewingRecipe {
                base: PotionType::HealingStrong,
                ingredient: 45,
                result: PotionType::HarmingStrong,
            },
            // Poison
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 44,
                result: PotionType::Poison,
            },
            BrewingRecipe {
                base: PotionType::Poison,
                ingredient: 25,
                result: PotionType::PoisonStrong,
            },
            BrewingRecipe {
                base: PotionType::Poison,
                ingredient: 45,
                result: PotionType::Harming,
            },
            // Regeneration
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 46,
                result: PotionType::Regeneration,
            },
            BrewingRecipe {
                base: PotionType::Regeneration,
                ingredient: 25,
                result: PotionType::RegenerationStrong,
            },
            // Fire Resistance
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 47,
                result: PotionType::FireResistance,
            },
            // Night Vision
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 53,
                result: PotionType::NightVision,
            },
            BrewingRecipe {
                base: PotionType::NightVision,
                ingredient: 20,
                result: PotionType::NightVisionLong,
            },
            BrewingRecipe {
                base: PotionType::NightVision,
                ingredient: 45,
                result: PotionType::Invisibility,
            },
            // Water Breathing
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 55,
                result: PotionType::WaterBreathing,
            },
            BrewingRecipe {
                base: PotionType::WaterBreathing,
                ingredient: 20,
                result: PotionType::WaterBreathingLong,
            },
            // Leaping
            BrewingRecipe {
                base: PotionType::Awkward,
                ingredient: 51,
                result: PotionType::Leaping,
            },
            BrewingRecipe {
                base: PotionType::Leaping,
                ingredient: 25,
                result: PotionType::LeapingStrong,
            },
            BrewingRecipe {
                base: PotionType::Leaping,
                ingredient: 45,
                result: PotionType::Slowness,
            },
            // Weakness
            BrewingRecipe {
                base: PotionType::Water,
                ingredient: 45,
                result: PotionType::Weakness,
            },
            BrewingRecipe {
                base: PotionType::Weakness,
                ingredient: 20,
                result: PotionType::WeaknessLong,
            },
        ];

        Self { recipes }
    }

    /// Resolves the output potion given a base potion and an ingredient item ID.
    #[must_use]
    pub fn brew(&self, base: PotionType, ingredient: u32) -> Option<PotionType> {
        self.recipes
            .iter()
            .find(|r| r.base == base && r.ingredient == ingredient)
            .map(|r| r.result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brewing_transitions() {
        let registry = BrewingRegistry::standard();

        // Water + Nether Wart (40) -> Awkward
        assert_eq!(
            registry.brew(PotionType::Water, 40),
            Some(PotionType::Awkward)
        );

        // Awkward + Sugar (42) -> Swiftness
        assert_eq!(
            registry.brew(PotionType::Awkward, 42),
            Some(PotionType::Swiftness)
        );

        // Swiftness + Glowstone (25) -> SwiftnessStrong
        assert_eq!(
            registry.brew(PotionType::Swiftness, 25),
            Some(PotionType::SwiftnessStrong)
        );

        // Swiftness + Fermented Spider Eye (45) -> Slowness
        assert_eq!(
            registry.brew(PotionType::Swiftness, 45),
            Some(PotionType::Slowness)
        );

        // Water + Fermented Spider Eye (45) -> Weakness
        assert_eq!(
            registry.brew(PotionType::Water, 45),
            Some(PotionType::Weakness)
        );

        // Weakness + Redstone (20) -> WeaknessLong
        assert_eq!(
            registry.brew(PotionType::Weakness, 20),
            Some(PotionType::WeaknessLong)
        );

        // Awkward + Gold Ingot (53) -> NightVision
        assert_eq!(
            registry.brew(PotionType::Awkward, 53),
            Some(PotionType::NightVision)
        );

        // NightVision + Redstone (20) -> NightVisionLong
        assert_eq!(
            registry.brew(PotionType::NightVision, 20),
            Some(PotionType::NightVisionLong)
        );

        // NightVision + Fermented Spider Eye (45) -> Invisibility
        assert_eq!(
            registry.brew(PotionType::NightVision, 45),
            Some(PotionType::Invisibility)
        );

        // Invalid ingredient -> None
        assert_eq!(registry.brew(PotionType::Water, 1), None);
    }

    #[test]
    fn test_potion_consumption_effects() {
        let swiftness = PotionType::SwiftnessStrong;
        let effect = swiftness.effect().unwrap();
        assert_eq!(effect.kind, StatusEffectKind::Speed);
        assert_eq!(effect.amplifier, 1);
        assert_eq!(effect.duration_ticks, 1800);

        let nv = PotionType::NightVisionLong;
        let nv_eff = nv.effect().unwrap();
        assert_eq!(nv_eff.kind, StatusEffectKind::NightVision);
        assert_eq!(nv_eff.duration_ticks, 9600);

        let leap = PotionType::LeapingStrong;
        let leap_eff = leap.effect().unwrap();
        assert_eq!(leap_eff.kind, StatusEffectKind::JumpBoost);
        assert_eq!(leap_eff.amplifier, 1);

        // Consumable item effects
        let spider_eye_effs = consume_item_effects(44).unwrap();
        assert_eq!(spider_eye_effs[0].kind, StatusEffectKind::Poison);
        assert_eq!(spider_eye_effs[0].duration_ticks, 80);

        let rotten_flesh_effs = consume_item_effects(48).unwrap();
        assert_eq!(rotten_flesh_effs[0].kind, StatusEffectKind::Slowness);
    }
}
