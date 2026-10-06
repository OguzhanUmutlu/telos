//! Tests for telos-core i18n and translation catalog.

use telos_core::i18n::{
    LanguageCatalog, Text, TextArg, format_pattern, language_display_name, parse_posix_locale,
};

#[test]
fn test_embedded_defaults_lookup() {
    let catalog = LanguageCatalog::with_default_embedded();

    assert_eq!(catalog.translate("menu.singleplayer"), "Singleplayer");
    assert_eq!(catalog.translate("menu.quit"), "Quit Game");
    assert_eq!(catalog.translate("generator.standard"), "Standard");
    assert_eq!(catalog.translate("gameMode.survival"), "Survival");

    // Spanish translation
    assert_eq!(
        catalog.translate_locale("es_es", "menu.singleplayer"),
        "Un jugador"
    );
    assert_eq!(
        catalog.translate_locale("es_es", "menu.quit"),
        "Salir del juego"
    );

    // Turkish translation
    assert_eq!(
        catalog.translate_locale("tr_tr", "menu.singleplayer"),
        "Tek Oyunculu"
    );
    assert_eq!(
        catalog.translate_locale("tr_tr", "menu.quit"),
        "Oyundan Çık"
    );
}

#[test]
fn test_locale_fallback_chain() {
    let mut catalog = LanguageCatalog::new();
    catalog.insert("en_us", "common.ok", "OK");
    catalog.insert("en_us", "common.only_en", "Only English");
    catalog.insert("es_es", "common.ok", "Aceptar");

    // Exact match in es_es
    assert_eq!(catalog.translate_locale("es_es", "common.ok"), "Aceptar");

    // Fallback to en_us
    assert_eq!(
        catalog.translate_locale("es_es", "common.only_en"),
        "Only English"
    );

    // Missing key entirely falls back to raw key
    assert_eq!(
        catalog.translate_locale("es_es", "nonexistent.key"),
        "nonexistent.key"
    );
}

#[test]
fn test_parameter_formatting_sequential() {
    let catalog = LanguageCatalog::new();

    // %s substitution
    let res = format_pattern(
        "Hello %s, welcome to %s!",
        &[TextArg::from("Alice"), TextArg::from("Telos")],
        &catalog,
        "en_us",
    );
    assert_eq!(res, "Hello Alice, welcome to Telos!");

    // %d and %s
    let res = format_pattern(
        "You have %d apples and %s",
        &[TextArg::from(42), TextArg::from("one pear")],
        &catalog,
        "en_us",
    );
    assert_eq!(res, "You have 42 apples and one pear");
}

#[test]
fn test_parameter_formatting_positional() {
    let catalog = LanguageCatalog::new();

    // %2$s then %1$s
    let res = format_pattern(
        "%2$s was placed by %1$s",
        &[TextArg::from("Player1"), TextArg::from("Stone Block")],
        &catalog,
        "en_us",
    );
    assert_eq!(res, "Stone Block was placed by Player1");
}

#[test]
fn test_percent_escape() {
    let catalog = LanguageCatalog::new();

    let res = format_pattern(
        "Progress: %d%% complete",
        &[TextArg::from(85)],
        &catalog,
        "en_us",
    );
    assert_eq!(res, "Progress: 85% complete");
}

#[test]
fn test_nested_text_components() {
    let mut catalog = LanguageCatalog::new();
    catalog.insert("en_us", "block.telos.stone", "Stone");
    catalog.insert("en_us", "action.mined", "Mined %s");

    let stone = Text::translatable("block.telos.stone");
    let mined = Text::translatable_with_args("action.mined", vec![TextArg::Text(Box::new(stone))]);

    assert_eq!(mined.resolve(&catalog, "en_us"), "Mined Stone");
}

#[test]
fn test_load_lang_json() {
    let mut catalog = LanguageCatalog::new();
    let json_data = r#"{
        "custom.title": "My Adventure",
        "custom.score": "Score: %d"
    }"#;

    let count = catalog
        .load_lang_json("en_us", json_data)
        .expect("Failed to parse JSON");
    assert_eq!(count, 2);

    assert_eq!(catalog.translate("custom.title"), "My Adventure");
    assert_eq!(
        catalog.format("custom.score", &[TextArg::from(100)]),
        "Score: 100"
    );
}

#[test]
fn test_posix_locale_parsing() {
    assert_eq!(parse_posix_locale("en_US.UTF-8"), "en_us");
    assert_eq!(parse_posix_locale("tr_TR.utf8"), "tr_tr");
    assert_eq!(parse_posix_locale("es_ES@euro"), "es_es");
    assert_eq!(parse_posix_locale("de-DE"), "de_de");
    assert_eq!(parse_posix_locale("fr"), "fr_fr");
    assert_eq!(parse_posix_locale("C"), "en_us");
    assert_eq!(parse_posix_locale(""), "en_us");
}

#[test]
fn test_language_display_names() {
    assert_eq!(language_display_name("en_us"), "English (US)");
    assert_eq!(language_display_name("tr_tr"), "Türkçe (Türkiye)");
    assert_eq!(language_display_name("es_es"), "Español (España)");
}
