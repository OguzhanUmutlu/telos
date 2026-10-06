//! Tests for telos-ui menu widgets and screens.

#![allow(clippy::float_cmp)]

use telos_ui::menu::widgets::{MenuButton, MenuSlider, MenuTextInput};
use telos_ui::menu::world_create::{WorldCreateAction, WorldCreateWizard};
use telos_ui::settings::GameSettings;

#[test]
fn test_settings_toml_roundtrip() {
    let mut original = GameSettings::default();
    original.video.view_distance = 16;
    original.video.fov = 85.0;
    original.audio.master_volume = 0.8;
    original.controls.mouse_sensitivity = 1.5;

    let toml_str = toml::to_string_pretty(&original).expect("Serialization failed");
    let deserialized: GameSettings = toml::from_str(&toml_str).expect("Deserialization failed");
    assert_eq!(original, deserialized);
}

#[test]
fn test_settings_clamping() {
    let mut settings = GameSettings::default();
    settings.video.view_distance = 100;
    settings.video.fov = 15.0;
    settings.audio.master_volume = -0.5;
    settings.controls.mouse_sensitivity = 50.0;

    settings.clamp();
    assert_eq!(settings.video.view_distance, 32);
    assert_eq!(settings.video.fov, 30.0);
    assert_eq!(settings.audio.master_volume, 0.0);
    assert_eq!(settings.controls.mouse_sensitivity, 3.0);
}

#[test]
fn test_menu_button_hit_test() {
    let mut btn = MenuButton::new(1, 100.0, 50.0, 200.0, 25.0, "Test Button");
    assert!(btn.contains(150.0, 60.0));
    assert!(btn.contains(100.0, 50.0));
    assert!(btn.contains(300.0, 75.0));
    assert!(!btn.contains(99.0, 60.0));
    assert!(!btn.contains(150.0, 76.0));

    btn.hovered = btn.contains(150.0, 60.0);
    assert!(btn.hovered);
}

#[test]
fn test_slider_scrubbing() {
    let mut slider = MenuSlider::new(
        1, 100.0, 50.0, 100.0, 20.0, "Volume", 0.0, 1.0, 0.5, "%", false,
    );
    assert_eq!(slider.value, 0.5);

    // Scrub to beginning
    slider.update_from_mouse_x(100.0);
    assert_eq!(slider.value, 0.0);

    // Scrub to end
    slider.update_from_mouse_x(200.0);
    assert_eq!(slider.value, 1.0);

    // Scrub to 25%
    slider.update_from_mouse_x(125.0);
    assert!((slider.value - 0.25).abs() < 0.05);
}

#[test]
fn test_text_input_editing() {
    let mut input = MenuTextInput::new(1, 0.0, 0.0, 100.0, 20.0, "Placeholder");
    input.focused = true;
    input.insert_char('H');
    input.insert_char('i');
    assert_eq!(input.text, "Hi");
    assert_eq!(input.cursor_pos, 2);

    input.backspace();
    assert_eq!(input.text, "H");
    assert_eq!(input.cursor_pos, 1);
}

#[test]
fn test_world_create_wizard_numeric_and_hashed_seed() {
    let mut wizard = WorldCreateWizard::new();
    wizard.update_layout(800.0, 600.0);
    wizard.name_input.text = "Alpha World".to_string();
    wizard.seed_input.text = "42".to_string();

    let temp_dir = tempfile::tempdir().expect("Failed to create tempdir");
    let action = wizard.handle_mouse_click(
        wizard.buttons[1].x + 5.0,
        wizard.buttons[1].y + 5.0,
        temp_dir.path(),
    );

    match action {
        Some(WorldCreateAction::CreateWorld {
            name,
            seed,
            generator,
            ..
        }) => {
            assert_eq!(name, "Alpha World");
            assert_eq!(seed, 42);
            assert_eq!(generator, "Standard");
        }
        other => panic!("Expected CreateWorld action, got {other:?}"),
    }

    // Now test string hash seed
    let mut wizard2 = WorldCreateWizard::new();
    wizard2.update_layout(800.0, 600.0);
    wizard2.name_input.text = "Beta World".to_string();
    wizard2.seed_input.text = "CustomSeedString".to_string();
    let action2 = wizard2.handle_mouse_click(
        wizard2.buttons[1].x + 5.0,
        wizard2.buttons[1].y + 5.0,
        temp_dir.path(),
    );
    match action2 {
        Some(WorldCreateAction::CreateWorld { seed, .. }) => {
            assert_ne!(seed, 0);
            assert_ne!(seed, 42);
        }
        other => panic!("Expected CreateWorld action, got {other:?}"),
    }
}
