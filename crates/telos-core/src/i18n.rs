//! High-performance, pure-CPU internationalization (i18n) and translation engine.
//!
//! Supports standard Minecraft-style `lang/<locale>.json` flat key-value files,
//! parameter formatting (`%s`, `%d`, positional `%1$s`, `%%`), fallback resolution
//! (`<locale>` -> `en_us` -> raw key), and system locale detection.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;
use thiserror::Error;

/// Errors arising during language file loading or translation parsing.
#[derive(Debug, Error)]
pub enum I18nError {
    /// Failed to read language file from disk.
    #[error("I/O error reading language file: {0}")]
    Io(#[from] std::io::Error),
    /// Failed to parse JSON dictionary.
    #[error("Failed to parse language JSON: {0}")]
    Json(#[from] serde_json::Error),
}

/// A formatted argument for translatable text templates.
#[derive(Debug, Clone, PartialEq)]
pub enum TextArg {
    /// String argument.
    String(String),
    /// Integer argument.
    Int(i64),
    /// Floating-point argument.
    Float(f64),
    /// Nested text component.
    Text(Box<Text>),
}

impl From<&str> for TextArg {
    fn from(s: &str) -> Self {
        Self::String(s.to_string())
    }
}

impl From<String> for TextArg {
    fn from(s: String) -> Self {
        Self::String(s)
    }
}

impl From<i32> for TextArg {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<i64> for TextArg {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<u32> for TextArg {
    fn from(v: u32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<usize> for TextArg {
    fn from(v: usize) -> Self {
        #[allow(clippy::cast_possible_wrap)]
        Self::Int(v as i64)
    }
}

impl From<f32> for TextArg {
    fn from(v: f32) -> Self {
        Self::Float(f64::from(v))
    }
}

impl From<f64> for TextArg {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<bool> for TextArg {
    fn from(b: bool) -> Self {
        Self::String(b.to_string())
    }
}

impl From<Text> for TextArg {
    fn from(t: Text) -> Self {
        Self::Text(Box::new(t))
    }
}

/// A translatable or literal text component.
#[derive(Debug, Clone, PartialEq)]
pub enum Text {
    /// Verbatim unlocalized text.
    Literal(String),
    /// Translatable key with optional interpolated arguments.
    Translatable {
        /// The lookup key (e.g. `menu.singleplayer`).
        key: String,
        /// Positional or ordered substitution arguments.
        args: Vec<TextArg>,
    },
}

impl Text {
    /// Creates a verbatim literal text element.
    #[must_use]
    pub fn literal(s: impl Into<String>) -> Self {
        Self::Literal(s.into())
    }

    /// Creates a translatable text element without arguments.
    #[must_use]
    pub fn translatable(key: impl Into<String>) -> Self {
        Self::Translatable {
            key: key.into(),
            args: Vec::new(),
        }
    }

    /// Creates a translatable text element with substitution arguments.
    #[must_use]
    pub fn translatable_with_args(key: impl Into<String>, args: Vec<TextArg>) -> Self {
        Self::Translatable {
            key: key.into(),
            args,
        }
    }

    /// Resolves this text component into a formatted string using the provided catalog.
    #[must_use]
    pub fn resolve(&self, catalog: &LanguageCatalog, locale: &str) -> String {
        match self {
            Self::Literal(s) => s.clone(),
            Self::Translatable { key, args } => {
                let pattern = catalog.translate_locale(locale, key);
                format_pattern(pattern, args, catalog, locale)
            }
        }
    }
}

impl fmt::Display for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(s) => write!(f, "{s}"),
            Self::Translatable { key, .. } => write!(f, "{key}"),
        }
    }
}

/// Formats a template string containing `%s`, `%d`, `%1$s`, or `%%` with arguments.
#[must_use]
pub fn format_pattern(
    pattern: &str,
    args: &[TextArg],
    catalog: &LanguageCatalog,
    locale: &str,
) -> String {
    let mut out = String::with_capacity(pattern.len() + args.len() * 8);
    let mut chars = pattern.chars().peekable();
    let mut sequential_idx = 0;

    while let Some(ch) = chars.next() {
        if ch != '%' {
            out.push(ch);
            continue;
        }

        // Handle escape: %% -> %
        if chars.peek() == Some(&'%') {
            chars.next();
            out.push('%');
            continue;
        }

        // Check for positional index: %1$s, %2$d, etc.
        let mut digits = String::new();
        while let Some(&d) = chars.peek() {
            if d.is_ascii_digit() {
                digits.push(d);
                chars.next();
            } else {
                break;
            }
        }

        let mut target_idx = None;
        if !digits.is_empty() && chars.peek() == Some(&'$') {
            chars.next(); // consume '$'
            if let Ok(num) = digits.parse::<usize>()
                && num > 0
            {
                target_idx = Some(num - 1);
            }
        } else if !digits.is_empty() {
            // Digits were not followed by '$' (e.g. %20s width format or plain text)
            out.push('%');
            out.push_str(&digits);
            continue;
        }

        let specifier = chars.next();
        let idx = target_idx.unwrap_or_else(|| {
            let cur = sequential_idx;
            sequential_idx += 1;
            cur
        });

        if let Some(arg) = args.get(idx) {
            match arg {
                TextArg::String(s) => out.push_str(s),
                TextArg::Int(v) => out.push_str(&v.to_string()),
                TextArg::Float(f) => {
                    use std::fmt::Write as _;
                    // Check if specifier is integer %d
                    if specifier == Some('d') {
                        let _ = write!(out, "{}", (*f).round() as i64);
                    } else {
                        let _ = write!(out, "{f:.2}");
                    }
                }
                TextArg::Text(t) => out.push_str(&t.resolve(catalog, locale)),
            }
        } else {
            // Missing argument: preserve token as placeholder
            out.push('%');
            if let Some(pos) = target_idx {
                out.push_str(&(pos + 1).to_string());
                out.push('$');
            }
            if let Some(sp) = specifier {
                out.push(sp);
            }
        }
    }

    out
}

/// Thread-safe in-memory internationalization catalog.
#[derive(Debug, Clone)]
pub struct LanguageCatalog {
    /// Maps locale (e.g. "`en_us`") to (key -> translated string).
    translations: HashMap<String, HashMap<String, String>>,
    /// Active selected locale.
    active_locale: String,
}

impl Default for LanguageCatalog {
    fn default() -> Self {
        Self::with_default_embedded()
    }
}

impl LanguageCatalog {
    /// Creates an empty language catalog with active locale `"en_us"`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            translations: HashMap::new(),
            active_locale: "en_us".to_string(),
        }
    }

    /// Creates a language catalog initialized with built-in embedded multilingual translations.
    #[must_use]
    pub fn with_default_embedded() -> Self {
        let mut catalog = Self::new();
        catalog.load_embedded_defaults();
        catalog
    }

    /// Returns the currently active locale string.
    #[must_use]
    pub fn active_locale(&self) -> &str {
        &self.active_locale
    }

    /// Sets the currently active locale string (e.g. `"en_us"` or `"tr_tr"`).
    pub fn set_active_locale(&mut self, locale: impl Into<String>) {
        self.active_locale = locale.into();
    }

    /// Returns a list of all currently loaded locale keys.
    #[must_use]
    pub fn available_locales(&self) -> Vec<String> {
        let mut list: Vec<String> = self.translations.keys().cloned().collect();
        if !list.contains(&"en_us".to_string()) {
            list.push("en_us".to_string());
        }
        list.sort();
        list
    }

    /// Returns true if the catalog contains translations for the specified locale.
    #[must_use]
    pub fn has_locale(&self, locale: &str) -> bool {
        self.translations.contains_key(locale)
    }

    /// Fast zero-allocation lookup resolving `locale` -> `"en_us"` -> None.
    #[must_use]
    pub fn get<'a>(&'a self, locale: &str, key: &str) -> Option<&'a str> {
        if let Some(loc_map) = self.translations.get(locale)
            && let Some(val) = loc_map.get(key)
        {
            return Some(val.as_str());
        }
        if locale != "en_us"
            && let Some(en_map) = self.translations.get("en_us")
            && let Some(val) = en_map.get(key)
        {
            return Some(val.as_str());
        }
        None
    }

    /// Translates a key using the active locale, falling back to `en_us` and finally to the key.
    #[must_use]
    pub fn translate<'a>(&'a self, key: &'a str) -> &'a str {
        self.get(&self.active_locale, key).unwrap_or(key)
    }

    /// Translates a key using an explicit locale.
    #[must_use]
    pub fn translate_locale<'a>(&'a self, locale: &str, key: &'a str) -> &'a str {
        self.get(locale, key).unwrap_or(key)
    }

    /// Formats a key with arguments using the active locale.
    #[must_use]
    pub fn format(&self, key: &str, args: &[TextArg]) -> String {
        self.format_locale(&self.active_locale, key, args)
    }

    /// Formats a key with arguments using an explicit locale.
    #[must_use]
    pub fn format_locale(&self, locale: &str, key: &str, args: &[TextArg]) -> String {
        let pattern = self.translate_locale(locale, key);
        format_pattern(pattern, args, self, locale)
    }

    /// Inserts a single translation entry into the catalog.
    pub fn insert(
        &mut self,
        locale: impl Into<String>,
        key: impl Into<String>,
        val: impl Into<String>,
    ) {
        self.translations
            .entry(locale.into())
            .or_default()
            .insert(key.into(), val.into());
    }

    /// Loads a flat JSON language map string into the specified locale.
    pub fn load_lang_json(&mut self, locale: &str, json_str: &str) -> Result<usize, I18nError> {
        let map: HashMap<String, String> = serde_json::from_str(json_str)?;
        let count = map.len();
        let target = self.translations.entry(locale.to_string()).or_default();
        for (k, v) in map {
            target.insert(k, v);
        }
        Ok(count)
    }

    /// Loads a language JSON file from a path.
    pub fn load_lang_file(&mut self, locale: &str, path: &Path) -> Result<usize, I18nError> {
        let content = std::fs::read_to_string(path)?;
        self.load_lang_json(locale, &content)
    }

    /// Scans a directory for all `<locale>.json` files and loads them.
    pub fn load_from_dir(&mut self, dir: &Path) -> Result<usize, I18nError> {
        if !dir.is_dir() {
            return Ok(0);
        }
        let mut total = 0;
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json")
                && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
            {
                let locale = stem.to_ascii_lowercase();
                if let Ok(count) = self.load_lang_file(&locale, &path) {
                    total += count;
                }
            }
        }
        Ok(total)
    }

    /// Populates built-in embedded multilingual fallback dictionaries.
    #[allow(clippy::too_many_lines, clippy::similar_names)]
    fn load_embedded_defaults(&mut self) {
        // 1. English (en_us) core defaults
        let en_pairs = [
            ("menu.singleplayer", "Singleplayer"),
            ("menu.multiplayer", "Multiplayer"),
            ("menu.options", "Options..."),
            ("menu.quit", "Quit Game"),
            ("menu.game", "Game Paused"),
            ("menu.returnToGame", "Back to Game"),
            ("menu.returnToMenu", "Save & Quit to Title"),
            ("selectWorld.title", "Select World"),
            ("selectWorld.select", "Play Selected World"),
            ("selectWorld.create", "Create New World"),
            ("selectWorld.delete", "Delete"),
            (
                "selectWorld.deleteQuestion",
                "Are you sure you want to delete this world?",
            ),
            ("selectWorld.enterName", "World Name"),
            ("selectWorld.enterSeed", "Seed (Leave blank for random)"),
            ("selectWorld.worldType", "World Type: %s"),
            ("generator.standard", "Standard"),
            ("generator.flat", "Superflat"),
            ("generator.void", "Void"),
            ("gameMode.survival", "Survival"),
            ("gameMode.creative", "Creative"),
            ("gameMode.spectator", "Spectator"),
            ("options.video", "Video Settings..."),
            ("options.audio", "Music & Sounds..."),
            ("options.controls", "Controls..."),
            ("options.language", "Language..."),
            ("options.tab.video", "Video"),
            ("options.tab.audio", "Audio"),
            ("options.tab.controls", "Controls"),
            ("options.tab.language", "Language"),
            ("options.fov", "FOV: %s"),
            ("options.fov.max", "FOV: Quake Pro"),
            ("options.renderDistance", "Render Distance: %s Chunks"),
            (
                "options.verticalRenderDistance",
                "Vertical Render Distance: %s Chunks",
            ),
            ("options.guiScale", "GUI Scale: %s"),
            ("options.guiScale.auto", "GUI Scale: Auto"),
            ("options.framerateLimit", "Max Framerate: %s fps"),
            ("options.framerateLimit.max", "Max Framerate: Unlimited"),
            ("options.vsync", "VSync: %s"),
            ("options.culling", "Hi-Z Occlusion Culling: %s"),
            ("options.sensitivity", "Mouse Sensitivity: %s"),
            ("options.invertMouse", "Invert Mouse Y: %s"),
            ("options.audio.master", "Master Volume: %s%"),
            ("options.audio.music", "Music: %s%"),
            ("options.audio.weather", "Weather: %s%"),
            ("options.audio.blocks", "Blocks: %s%"),
            ("options.audio.entities", "Entities: %s%"),
            ("options.on", "ON"),
            ("options.off", "OFF"),
            ("gui.done", "Done"),
            ("gui.cancel", "Cancel"),
            ("gui.back", "Back"),
            ("gui.yes", "Yes"),
            ("gui.no", "No"),
            ("chat.type.text", "<%s> %s"),
            ("chat.cannotSend", "Cannot send chat message"),
            ("block.telos.stone", "Stone"),
            ("block.telos.dirt", "Dirt"),
            ("block.telos.grass_block", "Grass Block"),
            ("block.telos.sand", "Sand"),
            ("block.telos.gravel", "Gravel"),
            ("block.telos.oak_planks", "Oak Planks"),
            ("block.telos.oak_log", "Oak Log"),
            ("block.telos.birch_log", "Birch Log"),
            ("block.telos.spruce_log", "Spruce Log"),
            ("block.telos.oak_leaves", "Oak Leaves"),
            ("block.telos.birch_leaves", "Birch Leaves"),
            ("block.telos.spruce_leaves", "Spruce Leaves"),
            ("block.telos.glass", "Glass"),
            ("block.telos.water", "Water"),
            ("block.telos.coal_ore", "Coal Ore"),
            ("block.telos.iron_ore", "Iron Ore"),
            ("block.telos.copper_ore", "Copper Ore"),
            ("block.telos.gold_ore", "Gold Ore"),
            ("block.telos.redstone_ore", "Redstone Ore"),
            ("block.telos.lapis_ore", "Lapis Ore"),
            ("block.telos.diamond_ore", "Diamond Ore"),
            ("block.telos.emerald_ore", "Emerald Ore"),
            ("block.telos.deepslate", "Deepslate"),
            ("block.telos.deepslate_coal_ore", "Deepslate Coal Ore"),
            ("block.telos.deepslate_iron_ore", "Deepslate Iron Ore"),
            ("block.telos.deepslate_copper_ore", "Deepslate Copper Ore"),
            ("block.telos.deepslate_gold_ore", "Deepslate Gold Ore"),
            (
                "block.telos.deepslate_redstone_ore",
                "Deepslate Redstone Ore",
            ),
            ("block.telos.deepslate_lapis_ore", "Deepslate Lapis Ore"),
            ("block.telos.deepslate_diamond_ore", "Deepslate Diamond Ore"),
            ("block.telos.deepslate_emerald_ore", "Deepslate Emerald Ore"),
            ("block.telos.granite", "Granite"),
            ("block.telos.diorite", "Diorite"),
            ("block.telos.andesite", "Andesite"),
            ("block.telos.tuff", "Tuff"),
            ("block.telos.raw_iron_block", "Block of Raw Iron"),
            ("block.telos.raw_copper_block", "Block of Raw Copper"),
            ("item.telos.wooden_pickaxe", "Wooden Pickaxe"),
            ("item.telos.stone_pickaxe", "Stone Pickaxe"),
            ("item.telos.iron_sword", "Iron Sword"),
            ("item.telos.rotten_flesh", "Rotten Flesh"),
            ("item.telos.porkchop", "Raw Porkchop"),
            ("item.telos.beef", "Raw Beef"),
            ("item.telos.leather", "Leather"),
            // Phase 49 Structure Blocks & Loot Items
            ("block.telos.mossy_cobblestone", "Mossy Cobblestone"),
            ("block.telos.monster_spawner", "Monster Spawner"),
            ("block.telos.chest", "Chest"),
            ("item.telos.mossy_cobblestone", "Mossy Cobblestone"),
            ("item.telos.monster_spawner", "Monster Spawner"),
            ("item.telos.chest", "Chest"),
            ("item.telos.iron_ingot", "Iron Ingot"),
            ("item.telos.gold_ingot", "Gold Ingot"),
            ("item.telos.coal", "Coal"),
            ("item.telos.string", "String"),
            ("item.telos.gunpowder", "Gunpowder"),
            ("item.telos.bread", "Bread"),
            ("item.telos.wheat", "Wheat"),
            ("item.telos.saddle", "Saddle"),
            ("item.telos.name_tag", "Name Tag"),
        ];

        for (k, v) in en_pairs {
            self.insert("en_us", k, v);
        }

        // 2. Spanish (es_es) core translations
        let es_pairs = [
            ("menu.singleplayer", "Un jugador"),
            ("menu.multiplayer", "Multijugador"),
            ("menu.options", "Opciones..."),
            ("menu.quit", "Salir del juego"),
            ("menu.game", "Juego pausado"),
            ("menu.returnToGame", "Volver al juego"),
            ("menu.returnToMenu", "Guardar y volver al título"),
            ("selectWorld.title", "Seleccionar mundo"),
            ("selectWorld.select", "Jugar al mundo seleccionado"),
            ("selectWorld.create", "Crear un mundo nuevo"),
            ("selectWorld.delete", "Eliminar"),
            ("selectWorld.enterName", "Nombre del mundo"),
            ("selectWorld.enterSeed", "Semilla (vacío para aleatoria)"),
            ("selectWorld.worldType", "Tipo de mundo: %s"),
            ("generator.standard", "Estándar"),
            ("generator.flat", "Superllano"),
            ("generator.void", "Vacío"),
            ("gameMode.survival", "Supervivencia"),
            ("gameMode.creative", "Creativo"),
            ("options.video", "Gráficos..."),
            ("options.audio", "Música y sonido..."),
            ("options.controls", "Controles..."),
            ("options.language", "Idioma..."),
            ("options.tab.video", "Vídeo"),
            ("options.tab.audio", "Sonido"),
            ("options.tab.controls", "Controles"),
            ("options.tab.language", "Idioma"),
            ("gui.done", "Hecho"),
            ("gui.cancel", "Cancelar"),
            ("gui.back", "Atrás"),
            ("options.on", "SÍ"),
            ("options.off", "NO"),
        ];
        for (k, v) in es_pairs {
            self.insert("es_es", k, v);
        }

        // 3. German (de_de) core translations
        let de_pairs = [
            ("menu.singleplayer", "Einzelspieler"),
            ("menu.multiplayer", "Mehrspieler"),
            ("menu.options", "Optionen..."),
            ("menu.quit", "Spiel beenden"),
            ("menu.game", "Spiel pausiert"),
            ("menu.returnToGame", "Zurück zum Spiel"),
            ("menu.returnToMenu", "Speichern und zum Hauptmenü"),
            ("selectWorld.title", "Welt auswählen"),
            ("selectWorld.select", "Ausgewählte Welt spielen"),
            ("selectWorld.create", "Neue Welt erstellen"),
            ("selectWorld.delete", "Löschen"),
            ("selectWorld.enterName", "Name der Welt"),
            ("selectWorld.enterSeed", "Startwert (leer für Zufall)"),
            ("selectWorld.worldType", "Welttyp: %s"),
            ("generator.standard", "Standard"),
            ("generator.flat", "Flachwelt"),
            ("generator.void", "Leere"),
            ("gameMode.survival", "Überleben"),
            ("gameMode.creative", "Kreativ"),
            ("options.video", "Grafikeinstellungen..."),
            ("options.audio", "Musik & Geräusche..."),
            ("options.controls", "Steuerung..."),
            ("options.language", "Sprache..."),
            ("options.tab.video", "Grafik"),
            ("options.tab.audio", "Audio"),
            ("options.tab.controls", "Steuerung"),
            ("options.tab.language", "Sprache"),
            ("gui.done", "Fertig"),
            ("gui.cancel", "Abbrechen"),
            ("gui.back", "Zurück"),
            ("options.on", "AN"),
            ("options.off", "AUS"),
        ];
        for (k, v) in de_pairs {
            self.insert("de_de", k, v);
        }

        // 4. French (fr_fr) core translations
        let fr_pairs = [
            ("menu.singleplayer", "Solo"),
            ("menu.multiplayer", "Multijoueur"),
            ("menu.options", "Options..."),
            ("menu.quit", "Quitter le jeu"),
            ("menu.game", "Partie en pause"),
            ("menu.returnToGame", "Reprendre la partie"),
            ("menu.returnToMenu", "Sauvegarder et quitter"),
            ("selectWorld.title", "Sélectionner un monde"),
            ("selectWorld.select", "Jouer dans ce monde"),
            ("selectWorld.create", "Créer un nouveau monde"),
            ("selectWorld.delete", "Supprimer"),
            ("selectWorld.enterName", "Nom du monde"),
            (
                "selectWorld.enterSeed",
                "Graine (laisser vide pour aléatoire)",
            ),
            ("selectWorld.worldType", "Type de monde : %s"),
            ("generator.standard", "Standard"),
            ("generator.flat", "Plat"),
            ("generator.void", "Vide"),
            ("gameMode.survival", "Survie"),
            ("gameMode.creative", "Créatif"),
            ("options.video", "Options graphiques..."),
            ("options.audio", "Musique et sons..."),
            ("options.controls", "Contrôles..."),
            ("options.language", "Langue..."),
            ("options.tab.video", "Graphismes"),
            ("options.tab.audio", "Audio"),
            ("options.tab.controls", "Contrôles"),
            ("options.tab.language", "Langue"),
            ("gui.done", "Terminé"),
            ("gui.cancel", "Annuler"),
            ("gui.back", "Retour"),
            ("options.on", "OUI"),
            ("options.off", "NON"),
        ];
        for (k, v) in fr_pairs {
            self.insert("fr_fr", k, v);
        }

        // 5. Turkish (tr_tr) core translations
        let tr_pairs = [
            ("menu.singleplayer", "Tek Oyunculu"),
            ("menu.multiplayer", "Çok Oyunculu"),
            ("menu.options", "Ayarlar..."),
            ("menu.quit", "Oyundan Çık"),
            ("menu.game", "Oyun Duraklatıldı"),
            ("menu.returnToGame", "Oyuna Dön"),
            ("menu.returnToMenu", "Kaydet ve Başlığa Dön"),
            ("selectWorld.title", "Dünya Seç"),
            ("selectWorld.select", "Seçilen Dünyayı Oyna"),
            ("selectWorld.create", "Yeni Dünya Yarat"),
            ("selectWorld.delete", "Sil"),
            ("selectWorld.enterName", "Dünya Adı"),
            ("selectWorld.enterSeed", "Tohum (Rastgele için boş bırakın)"),
            ("selectWorld.worldType", "Dünya Tipi: %s"),
            ("generator.standard", "Standart"),
            ("generator.flat", "Dümdüz"),
            ("generator.void", "Boşluk"),
            ("gameMode.survival", "Hayatta Kalma"),
            ("gameMode.creative", "Yaratıcı"),
            ("options.video", "Görüntü Ayarları..."),
            ("options.audio", "Müzik ve Sesler..."),
            ("options.controls", "Kontroller..."),
            ("options.language", "Dil..."),
            ("options.tab.video", "Görüntü"),
            ("options.tab.audio", "Ses"),
            ("options.tab.controls", "Kontroller"),
            ("options.tab.language", "Dil"),
            ("gui.done", "Tamam"),
            ("gui.cancel", "İptal"),
            ("gui.back", "Geri"),
            ("options.on", "AÇIK"),
            ("options.off", "KAPALI"),
        ];
        for (k, v) in tr_pairs {
            self.insert("tr_tr", k, v);
        }
    }
}

/// Returns the native user-facing display name of a locale code.
#[must_use]
pub fn language_display_name(locale: &str) -> &'static str {
    match locale {
        "en_us" => "English (US)",
        "es_es" => "Español (España)",
        "de_de" => "Deutsch (Deutschland)",
        "fr_fr" => "Français (France)",
        "tr_tr" => "Türkçe (Türkiye)",
        "it_it" => "Italiano",
        "pt_br" => "Português (Brasil)",
        "ru_ru" => "Русский",
        "zh_cn" => "简体中文",
        "ja_jp" => "日本語",
        _ => "Custom",
    }
}

/// Detects the system locale from environment variables (`LC_ALL`, `LC_MESSAGES`, `LANG`).
#[must_use]
pub fn detect_system_locale() -> String {
    let raw = std::env::var("LC_ALL")
        .or_else(|_| std::env::var("LC_MESSAGES"))
        .or_else(|_| std::env::var("LANG"))
        .unwrap_or_default();

    parse_posix_locale(&raw)
}

/// Normalizes a POSIX locale string (e.g. `en_US.UTF-8` or `tr_TR`) to `<lang>_<country>`.
#[must_use]
pub fn parse_posix_locale(raw: &str) -> String {
    if raw.is_empty() || raw == "C" || raw == "POSIX" {
        return "en_us".to_string();
    }

    // Strip encoding (e.g. .UTF-8) and modifier (@euro)
    let base = raw.split('.').next().unwrap_or(raw);
    let base = base.split('@').next().unwrap_or(base);

    let lower = base.to_ascii_lowercase().replace('-', "_");
    if lower.contains('_') {
        lower
    } else {
        // e.g. "en" -> "en_us", "tr" -> "tr_tr", "es" -> "es_es"
        match lower.as_str() {
            "en" => "en_us".to_string(),
            "es" => "es_es".to_string(),
            "de" => "de_de".to_string(),
            "fr" => "fr_fr".to_string(),
            "tr" => "tr_tr".to_string(),
            "it" => "it_it".to_string(),
            "pt" => "pt_br".to_string(),
            "ru" => "ru_ru".to_string(),
            "zh" => "zh_cn".to_string(),
            "ja" => "ja_jp".to_string(),
            other => format!("{other}_{other}"),
        }
    }
}
