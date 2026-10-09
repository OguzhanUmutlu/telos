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

#[test]
fn test_main_menu_i18n_translation() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::main_menu::MainMenuScreen;

    let mut catalog = LanguageCatalog::with_default_embedded();
    let mut menu = MainMenuScreen::new();

    // Default English
    menu.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(menu.buttons[0].label, "Singleplayer");
    assert_eq!(menu.buttons[1].label, "Multiplayer");
    assert_eq!(menu.buttons[2].label, "Options...");
    assert_eq!(menu.buttons[3].label, "Quit Game");

    // Turkish (tr_tr)
    catalog.set_active_locale("tr_tr");
    menu.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(menu.buttons[0].label, "Tek Oyunculu");
    assert_eq!(menu.buttons[1].label, "Çok Oyunculu");
    assert_eq!(menu.buttons[2].label, "Ayarlar...");
    assert_eq!(menu.buttons[3].label, "Oyundan Çık");

    // Spanish (es_es)
    catalog.set_active_locale("es_es");
    menu.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(menu.buttons[0].label, "Un jugador");
    assert_eq!(menu.buttons[1].label, "Multijugador");
    assert_eq!(menu.buttons[2].label, "Opciones...");
    assert_eq!(menu.buttons[3].label, "Salir del juego");

    // German (de_de)
    catalog.set_active_locale("de_de");
    menu.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(menu.buttons[0].label, "Einzelspieler");
    assert_eq!(menu.buttons[1].label, "Mehrspieler");
    assert_eq!(menu.buttons[2].label, "Optionen...");
    assert_eq!(menu.buttons[3].label, "Spiel beenden");

    // French (fr_fr)
    catalog.set_active_locale("fr_fr");
    menu.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(menu.buttons[0].label, "Solo");
    assert_eq!(menu.buttons[1].label, "Multijoueur");
    assert_eq!(menu.buttons[2].label, "Options...");
    assert_eq!(menu.buttons[3].label, "Quitter le jeu");
}

#[test]
fn test_pause_menu_i18n_translation() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::pause_menu::PauseMenuScreen;

    let mut catalog = LanguageCatalog::with_default_embedded();
    let mut pause = PauseMenuScreen::new();

    catalog.set_active_locale("tr_tr");
    pause.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(pause.title, "Oyun Duraklatıldı");
    assert_eq!(pause.buttons.len(), 5);
    assert_eq!(pause.buttons[0].label, "Oyuna Dön");
    assert_eq!(pause.buttons[1].label, "Gelişmeler");
    assert_eq!(pause.buttons[2].label, "Ayarlar...");
    assert_eq!(pause.buttons[3].label, "Yerel Ağda Paylaş / Davet Et");
    assert_eq!(pause.buttons[4].label, "Kaydet ve Başlığa Dön");

    catalog.set_active_locale("de_de");
    pause.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(pause.title, "Spiel pausiert");
    assert_eq!(pause.buttons.len(), 5);
    assert_eq!(pause.buttons[0].label, "Zurück zum Spiel");
    assert_eq!(pause.buttons[1].label, "Fortschritte");
    assert_eq!(pause.buttons[2].label, "Optionen...");
    assert_eq!(pause.buttons[3].label, "Im LAN öffnen / Einladen");
    assert_eq!(pause.buttons[4].label, "Speichern und zum Hauptmenü");
}

#[test]
fn test_settings_screen_language_tab_and_switching() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::settings_screen::{SettingsScreen, SettingsTab};

    let catalog = LanguageCatalog::with_default_embedded();
    let settings = GameSettings::default();
    let mut screen = SettingsScreen::new(settings);

    // Initial English layout
    screen.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(screen.tab_buttons.len(), 4);
    assert_eq!(screen.tab_buttons[0].label, "Video");
    assert_eq!(screen.tab_buttons[1].label, "Audio");
    assert_eq!(screen.tab_buttons[2].label, "Controls");
    assert_eq!(screen.tab_buttons[3].label, "Language");

    // Click on Language tab (button ID 104)
    let lang_tab_btn = screen.tab_buttons[3].clone();
    screen.handle_mouse_click_i18n(
        lang_tab_btn.x + 5.0,
        lang_tab_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert_eq!(screen.active_tab, SettingsTab::Language);
    assert_eq!(screen.toggle_buttons.len(), 5);

    // Language buttons: 301 = en_us, 302 = es_es, 303 = de_de, 304 = fr_fr, 305 = tr_tr
    // Click Turkish (button 305, last toggle button)
    let tr_btn = screen.toggle_buttons[4].clone();
    screen.handle_mouse_click_i18n(tr_btn.x + 5.0, tr_btn.y + 5.0, 800.0, 600.0, &catalog);
    assert_eq!(screen.settings.gameplay.language, "tr_tr");

    // Tabs should now be translated in Turkish
    assert_eq!(screen.tab_buttons[0].label, "Görüntü");
    assert_eq!(screen.tab_buttons[1].label, "Ses");
    assert_eq!(screen.tab_buttons[2].label, "Kontroller");
    assert_eq!(screen.tab_buttons[3].label, "Dil");
    assert_eq!(screen.done_button.label, "Tamam");
}

#[test]
fn test_world_create_wizard_i18n_translation() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::world_create::WorldCreateWizard;

    let mut catalog = LanguageCatalog::with_default_embedded();
    let mut wizard = WorldCreateWizard::new();

    catalog.set_active_locale("es_es");
    wizard.update_layout_i18n(800.0, 600.0, &catalog);
    assert_eq!(wizard.title, "Crear un mundo nuevo");
    assert_eq!(wizard.name_label, "Nombre del mundo:");
    assert_eq!(wizard.seed_label, "Semilla (vacío para aleatoria):");
    assert_eq!(wizard.buttons[0].label, "Tipo de mundo: Estándar");
    assert_eq!(wizard.buttons[1].label, "Crear un mundo nuevo");
    assert_eq!(wizard.buttons[2].label, "Cancelar");
}

#[test]
fn test_settings_screen_controls_rebinding_flow() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::keybinds::{InputKey, KeyAction};
    use telos_ui::menu::settings_screen::{SettingsScreen, SettingsTab};

    let catalog = LanguageCatalog::with_default_embedded();
    let settings = GameSettings::default();
    let mut screen = SettingsScreen::new(settings);

    screen.update_layout_i18n(800.0, 600.0, &catalog);

    // Switch to Controls tab (button ID 103)
    let controls_tab_btn = screen.tab_buttons[2].clone();
    screen.handle_mouse_click_i18n(
        controls_tab_btn.x + 5.0,
        controls_tab_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert_eq!(screen.active_tab, SettingsTab::Controls);
    assert!(!screen.keybind_buttons.is_empty());

    // Click Forward keybind button
    let (action, fwd_btn) = screen.keybind_buttons[0].clone();
    assert_eq!(action, KeyAction::Forward);
    screen.handle_mouse_click_i18n(fwd_btn.x + 5.0, fwd_btn.y + 5.0, 800.0, 600.0, &catalog);
    assert!(screen.is_rebinding());
    assert_eq!(screen.listening_action, Some(KeyAction::Forward));

    // Rebind to ArrowUp
    let changed = screen.handle_key_input(InputKey::ArrowUp);
    assert!(changed);
    assert!(!screen.is_rebinding());
    assert_eq!(
        screen.settings.controls.keybinds.get(KeyAction::Forward),
        InputKey::ArrowUp
    );

    // Reset defaults button
    screen.update_layout_i18n(800.0, 600.0, &catalog);
    let reset_btn = screen.reset_keybinds_button.clone();
    screen.handle_mouse_click_i18n(reset_btn.x + 5.0, reset_btn.y + 5.0, 800.0, 600.0, &catalog);
    assert_eq!(
        screen.settings.controls.keybinds.get(KeyAction::Forward),
        InputKey::KeyW
    );

    // Test mouse wheel scroll
    screen.handle_mouse_wheel(2.0);
    assert_eq!(screen.scroll_y, 0.0); // Clamped at 0
    screen.handle_mouse_wheel(-2.0);
    assert!(screen.scroll_y > 0.0);
}

#[test]
fn test_settings_screen_volumetric_clouds_toggle() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::settings_screen::{SettingsScreen, SettingsTab};

    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = SettingsScreen::new(GameSettings::default());
    screen.active_tab = SettingsTab::Video;
    screen.update_layout_i18n(800.0, 600.0, &catalog);

    assert!(screen.settings.video.volumetric_clouds);

    // Locate button 209 (Clouds toggle)
    let clouds_btn = screen
        .toggle_buttons
        .iter()
        .find(|b| b.id == 209)
        .expect("Clouds toggle button 209 not found")
        .clone();

    assert!(clouds_btn.label.contains("ON"));

    // Click to toggle off
    screen.handle_mouse_click_i18n(
        clouds_btn.x + 5.0,
        clouds_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert!(!screen.settings.video.volumetric_clouds);

    // Re-check layout
    let updated_btn = screen
        .toggle_buttons
        .iter()
        .find(|b| b.id == 209)
        .expect("Clouds toggle button 209 not found")
        .clone();
    assert!(updated_btn.label.contains("OFF"));

    // Click to toggle back on
    screen.handle_mouse_click_i18n(
        updated_btn.x + 5.0,
        updated_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert!(screen.settings.video.volumetric_clouds);
}

#[test]
fn test_settings_screen_shadows_toggle() {
    use telos_core::i18n::LanguageCatalog;
    use telos_ui::menu::settings_screen::{SettingsScreen, SettingsTab};

    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = SettingsScreen::new(GameSettings::default());
    screen.active_tab = SettingsTab::Video;
    screen.update_layout_i18n(800.0, 600.0, &catalog);

    assert!(screen.settings.video.shadows);

    // Locate button 212 (Shadows toggle)
    let shadows_btn = screen
        .toggle_buttons
        .iter()
        .find(|b| b.id == 212)
        .expect("Shadows toggle button 212 not found")
        .clone();

    assert!(shadows_btn.label.contains("ON"));

    // Click to toggle off
    screen.handle_mouse_click_i18n(
        shadows_btn.x + 5.0,
        shadows_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert!(!screen.settings.video.shadows);

    // Re-check layout
    let updated_btn = screen
        .toggle_buttons
        .iter()
        .find(|b| b.id == 212)
        .expect("Shadows toggle button 212 not found")
        .clone();
    assert!(updated_btn.label.contains("OFF"));

    // Click to toggle back on
    screen.handle_mouse_click_i18n(
        updated_btn.x + 5.0,
        updated_btn.y + 5.0,
        800.0,
        600.0,
        &catalog,
    );
    assert!(screen.settings.video.shadows);
}
