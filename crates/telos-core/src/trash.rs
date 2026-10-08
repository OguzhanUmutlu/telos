//! Standards-compliant safe desktop trash and world deletion handling.
//!
//! Provides `FreeDesktop.org` XDG Trash specification compliance for moving saved
//! worlds and user files to the system trash (with `.trashinfo` metadata for restoration),
//! fallback to native desktop `gio trash`, and a local safety archive fallback
//! to ensure user data is never destroyed unrecoverably.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

/// Formats a `SystemTime` as an ISO 8601 / RFC 3339 UTC timestamp: `YYYY-MM-DDTHH:MM:SS`.
#[must_use]
pub fn format_iso8601_utc(time: SystemTime) -> String {
    let secs = time
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let sec_of_day = secs % 86400;
    let hour = sec_of_day / 3600;
    let min = (sec_of_day % 3600) / 60;
    let sec = sec_of_day % 60;

    // Howard Hinnant's algorithm for Gregorian date from days since 1970-01-01
    #[allow(clippy::cast_possible_wrap)]
    let days = (secs / 86400) as i64 + 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let doe = (days - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    #[allow(clippy::cast_possible_wrap)]
    let y = i64::from(yoe) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{min:02}:{sec:02}")
}

/// Percent-encodes an absolute path according to RFC 2396 / RFC 3986 as required
/// by the `FreeDesktop.org` Trash specification.
#[must_use]
pub fn percent_encode_path(path: &Path) -> String {
    let bytes = path.as_os_str().as_encoded_bytes();
    let mut out = String::with_capacity(bytes.len() + 8);
    for &b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~' | b'/') {
            out.push(b as char);
        } else {
            use std::fmt::Write;
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// Recursively copies all entries from `src` into `dst`.
fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ft = entry.file_type()?;
        let dest_child = dst.join(entry.file_name());
        if ft.is_dir() {
            copy_dir_all(&entry.path(), &dest_child)?;
        } else {
            fs::copy(entry.path(), dest_child)?;
        }
    }
    Ok(())
}

/// Safely moves a file or directory across filesystems if necessary.
fn move_path(src: &Path, dst: &Path) -> io::Result<()> {
    if fs::rename(src, dst).is_ok() {
        return Ok(());
    }
    if src.is_dir() {
        copy_dir_all(src, dst)?;
        fs::remove_dir_all(src)?;
    } else {
        fs::copy(src, dst)?;
        fs::remove_file(src)?;
    }
    Ok(())
}

/// Moves a target path into a `FreeDesktop.org`-compliant XDG Trash directory
/// (usually `$XDG_DATA_HOME/Trash` or `~/.local/share/Trash`).
///
/// Creates both the destination file/directory in `<trash_base>/files/` and
/// the corresponding restoration info metadata in `<trash_base>/info/<name>.trashinfo`.
/// Returns the final destination path in the trash.
pub fn move_to_xdg_trash(path: &Path, trash_base: &Path) -> io::Result<PathBuf> {
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Path does not exist: {}", path.display()),
        ));
    }

    let abs_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };

    let files_dir = trash_base.join("files");
    let info_dir = trash_base.join("info");
    fs::create_dir_all(&files_dir)?;
    fs::create_dir_all(&info_dir)?;

    let base_name = path
        .file_name()
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Cannot trash root or empty path",
            )
        })?
        .to_string_lossy()
        .to_string();

    // Disambiguate collision in trash
    let mut candidate_name = base_name.clone();
    let mut counter = 1u32;
    loop {
        let candidate_dest = files_dir.join(&candidate_name);
        let candidate_info = info_dir.join(format!("{candidate_name}.trashinfo"));
        if !candidate_dest.exists() && !candidate_info.exists() {
            break;
        }
        candidate_name = format!("{base_name}.{counter}");
        counter += 1;
    }

    let final_dest = files_dir.join(&candidate_name);
    let final_info = info_dir.join(format!("{candidate_name}.trashinfo"));

    // Per FreeDesktop spec: create the .trashinfo file FIRST to avoid race conditions
    let deletion_date = format_iso8601_utc(SystemTime::now());
    let encoded_path = percent_encode_path(&abs_path);
    let info_content = format!("[Trash Info]\nPath={encoded_path}\nDeletionDate={deletion_date}\n");

    fs::write(&final_info, info_content)?;

    // Move the actual file/directory
    if let Err(err) = move_path(path, &final_dest) {
        // Rollback info file on error
        let _ = fs::remove_file(&final_info);
        return Err(err);
    }

    Ok(final_dest)
}

/// Fallback mechanism: moves a file or directory into a hidden `.trash` directory
/// adjacent to the item's parent directory, stamped with a timestamp.
pub fn move_to_local_trash(path: &Path) -> io::Result<PathBuf> {
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Path does not exist: {}", path.display()),
        ));
    }

    let parent = path.parent().unwrap_or(path);
    let archive_dir = parent.join(".trash");
    fs::create_dir_all(&archive_dir)?;

    let file_name = path
        .file_name()
        .map_or("world", |n| n.to_str().unwrap_or("world"));

    let timestamp = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut dest_name = format!("{file_name}_{timestamp}");
    let mut counter = 1u32;
    loop {
        let target = archive_dir.join(&dest_name);
        if !target.exists() {
            break;
        }
        dest_name = format!("{file_name}_{timestamp}_{counter}");
        counter += 1;
    }

    let dest_path = archive_dir.join(&dest_name);
    move_path(path, &dest_path)?;
    Ok(dest_path)
}

/// Safely moves a world folder or file to the system trash.
///
/// Uses a robust 3-tier strategy:
/// 1. Standards-compliant `FreeDesktop` XDG Trash in `$XDG_DATA_HOME/Trash` or `~/.local/share/Trash`.
/// 2. Native desktop `gio trash` command if available.
/// 3. Adjacent `.trash` archive folder if system trash is unwritable or unavailable.
///
/// Ensures user save data is never deleted permanently without a recovery path.
pub fn move_to_trash(path: &Path) -> io::Result<PathBuf> {
    if !path.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Path does not exist: {}", path.display()),
        ));
    }

    // Tier 1: Try XDG Trash
    let xdg_trash_base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(dirs::data_local_dir)
        .or_else(|| dirs::home_dir().map(|h| h.join(".local/share")))
        .map(|base| base.join("Trash"));

    if let Some(trash_base) = xdg_trash_base
        && let Ok(dest) = move_to_xdg_trash(path, &trash_base)
    {
        return Ok(dest);
    }

    // Tier 2: Try native `gio trash`
    if let Ok(status) = std::process::Command::new("gio")
        .arg("trash")
        .arg(path)
        .status()
        && status.success()
        && !path.exists()
    {
        return Ok(PathBuf::from("trash:///").join(path.file_name().unwrap_or_default()));
    }

    // Tier 3: Local safe archive `.trash`
    move_to_local_trash(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::duration_suboptimal_units)]
    fn test_format_iso8601_utc() {
        // Unix epoch: 1970-01-01T00:00:00
        let epoch = SystemTime::UNIX_EPOCH;
        assert_eq!(format_iso8601_utc(epoch), "1970-01-01T00:00:00");

        // 2026-10-08 23:05:00 UTC (1791500700 secs)
        let t = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_791_500_700);
        let formatted = format_iso8601_utc(t);
        assert_eq!(formatted, "2026-10-08T23:05:00");
    }

    #[test]
    fn test_percent_encode_path() {
        let p = Path::new("/home/user/My Worlds/test_world (1)");
        let enc = percent_encode_path(p);
        assert!(enc.contains("/home/user/My%20Worlds/test_world%20%281%29"));
    }

    #[test]
    fn test_move_to_xdg_trash_directory() {
        let temp_dir = tempfile::tempdir().unwrap();
        let world_path = temp_dir.path().join("world_test_alpha");
        fs::create_dir_all(&world_path).unwrap();
        fs::write(world_path.join("world.toml"), "name = \"Alpha\"\n").unwrap();
        fs::write(world_path.join("level.dat"), b"voxel save").unwrap();

        let trash_dir = temp_dir.path().join("mock_trash");
        let dest = move_to_xdg_trash(&world_path, &trash_dir).unwrap();

        // Source world should no longer exist
        assert!(!world_path.exists());
        // Destination should exist in files/
        assert!(dest.exists());
        assert!(dest.join("world.toml").exists());
        assert!(dest.join("level.dat").exists());

        // Info file should exist in info/
        let info_file = trash_dir.join("info/world_test_alpha.trashinfo");
        assert!(info_file.exists());
        let info_content = fs::read_to_string(&info_file).unwrap();
        assert!(info_content.contains("[Trash Info]"));
        assert!(info_content.contains("Path="));
        assert!(info_content.contains("world_test_alpha"));
        assert!(info_content.contains("DeletionDate="));
    }

    #[test]
    fn test_move_to_xdg_trash_collision_disambiguation() {
        let temp_dir = tempfile::tempdir().unwrap();
        let trash_dir = temp_dir.path().join("mock_trash");

        // First item
        let world1 = temp_dir.path().join("collision_world");
        fs::create_dir_all(&world1).unwrap();
        fs::write(world1.join("meta.txt"), "v1").unwrap();
        let dest1 = move_to_xdg_trash(&world1, &trash_dir).unwrap();
        assert_eq!(dest1.file_name().unwrap(), "collision_world");

        // Second item with identical name
        let world2 = temp_dir.path().join("collision_world");
        fs::create_dir_all(&world2).unwrap();
        fs::write(world2.join("meta.txt"), "v2").unwrap();
        let dest2 = move_to_xdg_trash(&world2, &trash_dir).unwrap();
        assert_eq!(dest2.file_name().unwrap(), "collision_world.1");

        assert!(trash_dir.join("info/collision_world.trashinfo").exists());
        assert!(trash_dir.join("info/collision_world.1.trashinfo").exists());
    }

    #[test]
    fn test_move_to_local_trash_fallback() {
        let temp_dir = tempfile::tempdir().unwrap();
        let world = temp_dir.path().join("local_fallback_world");
        fs::create_dir_all(&world).unwrap();
        fs::write(world.join("save.dat"), "123").unwrap();

        let archived = move_to_local_trash(&world).unwrap();
        assert!(!world.exists());
        assert!(archived.exists());
        assert!(archived.parent().unwrap().ends_with(".trash"));
        assert!(archived.join("save.dat").exists());
    }

    #[test]
    fn test_move_nonexistent_path_fails() {
        let temp_dir = tempfile::tempdir().unwrap();
        let missing = temp_dir.path().join("does_not_exist");
        assert!(move_to_trash(&missing).is_err());
    }
}
