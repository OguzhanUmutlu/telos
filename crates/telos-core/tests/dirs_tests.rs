//! Unit tests for platform-native application directory resolution.

use telos_core::dirs::AppDirs;
use tempfile::tempdir;

#[test]
fn test_standard_dirs_resolution() {
    let dirs = AppDirs::standard();
    assert!(!dirs.is_portable());
    assert!(!dirs.data_dir().as_os_str().is_empty());
    assert!(!dirs.config_dir().as_os_str().is_empty());
    assert!(!dirs.cache_dir().as_os_str().is_empty());
    assert!(!dirs.logs_dir().as_os_str().is_empty());

    // On Linux/Unix, paths should end in 'telos'
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        assert!(dirs.data_dir().ends_with("telos"));
        assert!(dirs.config_dir().ends_with("telos"));
        assert!(dirs.cache_dir().ends_with("telos"));
        assert!(dirs.logs_dir().ends_with("logs"));
    }
}

#[test]
fn test_portable_dirs_isolation() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("portable_test");
    let dirs = AppDirs::portable(&base);

    assert!(dirs.is_portable());
    assert_eq!(dirs.data_dir(), base.join("data"));
    assert_eq!(dirs.config_dir(), base.join("config"));
    assert_eq!(dirs.cache_dir(), base.join("cache"));
    assert_eq!(dirs.logs_dir(), base.join("logs"));
    assert_eq!(dirs.saves_dir(), base.join("data").join("worlds"));
    assert_eq!(
        dirs.world_save_dir("arena"),
        base.join("data").join("worlds").join("arena")
    );
}

#[test]
fn test_from_data_dir() {
    let tmp = tempdir().unwrap();
    let custom = tmp.path().join("custom_root");
    let dirs = AppDirs::from_data_dir(&custom);

    assert!(!dirs.is_portable());
    assert_eq!(dirs.data_dir(), custom);
    assert_eq!(dirs.config_dir(), custom.join("config"));
    assert_eq!(dirs.cache_dir(), custom.join("cache"));
    assert_eq!(dirs.logs_dir(), custom.join("logs"));
    assert_eq!(dirs.saves_dir(), custom.join("worlds"));
}

#[test]
fn test_helper_subpaths() {
    let tmp = tempdir().unwrap();
    let dirs = AppDirs::portable(tmp.path());

    assert_eq!(
        dirs.config_file("settings.toml"),
        tmp.path().join("config").join("settings.toml")
    );
    assert_eq!(
        dirs.lod_cache_dir("server1"),
        tmp.path().join("cache").join("lod").join("server1")
    );
    assert_eq!(
        dirs.shader_cache_dir(),
        tmp.path().join("cache").join("shaders")
    );
    assert_eq!(dirs.assets_dir(), tmp.path().join("data").join("assets"));
    assert_eq!(dirs.mods_dir(), tmp.path().join("data").join("mods"));
}

#[test]
fn test_builder_overrides() {
    let tmp = tempdir().unwrap();
    let custom_cfg = tmp.path().join("my_config");
    let custom_cache = tmp.path().join("my_cache");

    let dirs = AppDirs::standard()
        .with_config_dir(&custom_cfg)
        .with_cache_dir(&custom_cache);

    assert_eq!(dirs.config_dir(), custom_cfg);
    assert_eq!(dirs.cache_dir(), custom_cache);
}

#[test]
fn test_ensure_dirs_exist() {
    let tmp = tempdir().unwrap();
    let base = tmp.path().join("ensure_test");
    let dirs = AppDirs::portable(&base);

    assert!(!dirs.data_dir().exists());
    assert!(!dirs.config_dir().exists());
    assert!(!dirs.cache_dir().exists());
    assert!(!dirs.logs_dir().exists());
    assert!(!dirs.saves_dir().exists());

    dirs.ensure_dirs_exist()
        .expect("ensure_dirs_exist must succeed");

    assert!(dirs.data_dir().is_dir());
    assert!(dirs.config_dir().is_dir());
    assert!(dirs.cache_dir().is_dir());
    assert!(dirs.logs_dir().is_dir());
    assert!(dirs.saves_dir().is_dir());
}
