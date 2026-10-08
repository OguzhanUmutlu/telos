//! Comprehensive integration tests for the modernized `WorldSelectScreen`,
//! confirmation prompts, and safe trash deletion workflows.

#![allow(clippy::float_cmp, clippy::duration_suboptimal_units)]

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};

use telos_core::i18n::LanguageCatalog;
use telos_core::trash::move_to_xdg_trash;
use telos_ui::font::BitmapFont;
use telos_ui::menu::widgets::ButtonStyle;
use telos_ui::menu::world_select::{WorldEntry, WorldSelectAction, WorldSelectScreen};

#[test]
fn test_world_select_initialization_and_empty_state() {
    let screen = WorldSelectScreen::new();
    assert!(screen.worlds.is_empty());
    assert_eq!(screen.selected_index, None);
    assert_eq!(screen.hovered_card_index, None);
    assert!(!screen.has_prompt());
    assert_eq!(screen.scroll_offset, 0.0);
    assert_eq!(screen.max_scroll, 0.0);
}

#[test]
fn test_world_select_card_rendering_and_metadata() {
    let font = BitmapFont::new_fallback(0);
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    screen.worlds.push(WorldEntry::new(
        "Survival Alpha",
        "save_alpha",
        PathBuf::from("/tmp/save_alpha"),
        1337,
        "Standard",
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_791_500_000)),
        4096,
    ));
    screen.worlds.push(WorldEntry::new(
        "Creative Flat",
        "save_flat",
        PathBuf::from("/tmp/save_flat"),
        42,
        "Flat",
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(1_791_400_000)),
        1024 * 1024 * 5,
    ));

    screen.selected_index = Some(0);
    screen.update_layout_i18n(640.0, 480.0, &catalog);

    let mut quads = Vec::new();
    screen.render(&font, 640.0, 480.0, 2, &mut quads);

    // Ensure rendering output produces quads for background, cards, header, and buttons
    assert!(!quads.is_empty());
    assert!(quads.len() > 30);
    assert_eq!(screen.worlds[0].formatted_size, "4.0 KB");
    assert_eq!(screen.worlds[1].formatted_size, "5.0 MB");
}

#[test]
fn test_world_select_gui_scaling_matrix() {
    let font = BitmapFont::new_fallback(0);
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    screen.worlds.push(WorldEntry::new(
        "Scale Test World",
        "scale_dir",
        PathBuf::from("/tmp/scale_dir"),
        100,
        "Standard",
        None,
        512,
    ));
    screen.selected_index = Some(0);

    for scale in [1u32, 2u32, 3u32, 4u32] {
        screen.update_layout_i18n(800.0, 600.0, &catalog);
        let mut quads = Vec::new();
        screen.render(&font, 800.0, 600.0, scale, &mut quads);
        assert!(!quads.is_empty(), "Failed rendering for scale {scale}");
    }
}

#[test]
fn test_world_select_card_selection_and_highlighting() {
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    for i in 0..3 {
        screen.worlds.push(WorldEntry::new(
            format!("World {i}"),
            format!("dir_{i}"),
            PathBuf::from(format!("/tmp/dir_{i}")),
            i as u64,
            "Standard",
            None,
            100,
        ));
    }

    screen.update_layout_i18n(640.0, 480.0, &catalog);
    screen.selected_index = Some(0);

    // Hover over second card
    let card_w = (640.0f32 - 80.0).clamp(340.0, 520.0);
    let card_h = 52.0;
    let card_gap = 8.0;
    let card_x = (640.0 - card_w) * 0.5;
    let card1_y = 50.0 + 1.0 * (card_h + card_gap);

    screen.handle_mouse_move(card_x + 10.0, card1_y + 10.0);
    assert_eq!(screen.hovered_card_index, Some(1));

    // Click to select second card
    let action =
        screen.handle_mouse_click_i18n(card_x + 10.0, card1_y + 10.0, 640.0, 480.0, &catalog);
    assert_eq!(action, None);
    assert_eq!(screen.selected_index, Some(1));
}

#[test]
fn test_world_select_double_click_play() {
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    let target_world = WorldEntry::new(
        "Play World",
        "play_dir",
        PathBuf::from("/tmp/play_dir"),
        999,
        "Standard",
        None,
        100,
    );
    screen.worlds.push(target_world.clone());
    screen.update_layout_i18n(640.0, 480.0, &catalog);

    let card_w = (640.0f32 - 80.0).clamp(340.0, 520.0);
    let card_x = (640.0 - card_w) * 0.5;
    let card_y = 50.0;

    // First click: selects card, does not play
    let a1 = screen.handle_mouse_click_i18n(card_x + 10.0, card_y + 10.0, 640.0, 480.0, &catalog);
    assert_eq!(a1, None);
    assert_eq!(screen.selected_index, Some(0));

    // Immediate second click on same card (< 400ms): triggers PlayWorld
    let a2 = screen.handle_mouse_click_i18n(card_x + 10.0, card_y + 10.0, 640.0, 480.0, &catalog);
    assert_eq!(a2, Some(WorldSelectAction::PlayWorld(target_world)));
}

#[test]
fn test_world_select_bottom_buttons_interactions() {
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    let entry = WorldEntry::new(
        "My World",
        "my_world",
        PathBuf::from("/tmp/my_world"),
        77,
        "Standard",
        None,
        200,
    );
    screen.worlds.push(entry.clone());
    screen.selected_index = Some(0);
    screen.update_layout_i18n(640.0, 480.0, &catalog);

    // Button 1: Play Selected World
    let play_btn = &screen.buttons[0];
    let play_act =
        screen.handle_mouse_click_i18n(play_btn.x + 2.0, play_btn.y + 2.0, 640.0, 480.0, &catalog);
    assert_eq!(play_act, Some(WorldSelectAction::PlayWorld(entry.clone())));

    // Button 2: Create New World
    let create_btn = &screen.buttons[1];
    let create_act = screen.handle_mouse_click_i18n(
        create_btn.x + 2.0,
        create_btn.y + 2.0,
        640.0,
        480.0,
        &catalog,
    );
    assert_eq!(create_act, Some(WorldSelectAction::CreateNewWorld));

    // Button 4: Back / Cancel
    let back_btn = &screen.buttons[3];
    let back_act =
        screen.handle_mouse_click_i18n(back_btn.x + 2.0, back_btn.y + 2.0, 640.0, 480.0, &catalog);
    assert_eq!(back_act, Some(WorldSelectAction::BackToTitle));

    // Button 3: Delete World -> Must open modal prompt, NOT return DeleteWorld immediately!
    let del_btn = &screen.buttons[2];
    let del_act =
        screen.handle_mouse_click_i18n(del_btn.x + 2.0, del_btn.y + 2.0, 640.0, 480.0, &catalog);
    assert_eq!(del_act, None);
    assert!(screen.has_prompt());
}

#[test]
fn test_delete_modal_prompt_ui_elements_and_rendering() {
    let font = BitmapFont::new_fallback(0);
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    let entry = WorldEntry::new(
        "Important Save",
        "save_dir",
        PathBuf::from("/tmp/save_dir"),
        42,
        "Standard",
        None,
        4096,
    );
    screen.worlds.push(entry);
    screen.selected_index = Some(0);
    screen.update_layout_i18n(640.0, 480.0, &catalog);

    // Open prompt
    assert!(screen.open_delete_prompt_i18n(&catalog));
    assert!(screen.has_prompt());

    let prompt = screen.delete_prompt.as_ref().unwrap();
    assert_eq!(prompt.confirm_button.style, ButtonStyle::Danger);
    assert_eq!(prompt.cancel_button.style, ButtonStyle::Default);

    // Hover confirm button
    screen.handle_mouse_move(prompt.confirm_button.x + 2.0, prompt.confirm_button.y + 2.0);
    assert!(screen.is_hovered());
    assert!(
        screen
            .delete_prompt
            .as_ref()
            .unwrap()
            .confirm_button
            .hovered
    );
    assert!(!screen.delete_prompt.as_ref().unwrap().cancel_button.hovered);

    // Render with prompt active
    let mut quads = Vec::new();
    screen.render(&font, 640.0, 480.0, 2, &mut quads);
    assert!(!quads.is_empty());
}

#[test]
fn test_delete_modal_confirm_and_trash_filesystem_integration() {
    let catalog = LanguageCatalog::with_default_embedded();
    let temp = tempfile::tempdir().unwrap();
    let worlds_dir = temp.path().join("worlds");
    fs::create_dir_all(&worlds_dir).unwrap();

    let world_folder = worlds_dir.join("real_world");
    fs::create_dir_all(&world_folder).unwrap();
    fs::write(world_folder.join("world.toml"), "name = \"Real World\"\n").unwrap();
    fs::write(world_folder.join("chunks.dat"), b"chunk voxels").unwrap();

    let mut screen = WorldSelectScreen::new();
    screen.scan_worlds(&worlds_dir);
    assert_eq!(screen.worlds.len(), 1);

    screen.update_layout_i18n(640.0, 480.0, &catalog);
    screen.open_delete_prompt_i18n(&catalog);

    // Confirm deletion
    let confirm_btn = screen
        .delete_prompt
        .as_ref()
        .unwrap()
        .confirm_button
        .clone();
    let action = screen.handle_mouse_click_i18n(
        confirm_btn.x + 2.0,
        confirm_btn.y + 2.0,
        640.0,
        480.0,
        &catalog,
    );

    let Some(WorldSelectAction::DeleteWorld(target_entry)) = action else {
        panic!("Expected DeleteWorld action");
    };
    assert!(!screen.has_prompt());

    // Execute safe trash operation
    let mock_trash = temp.path().join("system_trash");
    let trashed_dest = move_to_xdg_trash(&target_entry.path, &mock_trash).unwrap();

    // Original world must be gone from worlds directory
    assert!(!world_folder.exists());

    // World must now exist in trash files directory
    assert!(trashed_dest.exists());
    assert!(trashed_dest.join("world.toml").exists());
    assert!(trashed_dest.join("chunks.dat").exists());

    // Trashinfo file must exist with valid headers
    let info_path = mock_trash.join("info/real_world.trashinfo");
    assert!(info_path.exists());
    let info_content = fs::read_to_string(info_path).unwrap();
    assert!(info_content.contains("[Trash Info]"));
    assert!(info_content.contains("Path="));
    assert!(info_content.contains("DeletionDate="));

    // Rescan worlds: list is now empty!
    screen.scan_worlds(&worlds_dir);
    assert!(screen.worlds.is_empty());
}

#[test]
fn test_delete_modal_cancel_and_outside_clicks() {
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    screen.worlds.push(WorldEntry::new(
        "Preserved World",
        "p_dir",
        PathBuf::from("/tmp/p_dir"),
        1,
        "Standard",
        None,
        100,
    ));
    screen.selected_index = Some(0);
    screen.update_layout_i18n(640.0, 480.0, &catalog);

    // Open prompt
    screen.open_delete_prompt_i18n(&catalog);
    assert!(screen.has_prompt());

    // Click outside modal dialog window (e.g. top-left corner)
    let outside_act = screen.handle_mouse_click_i18n(5.0, 5.0, 640.0, 480.0, &catalog);
    assert_eq!(outside_act, None);
    assert!(!screen.has_prompt(), "Click outside should dismiss modal");

    // Open prompt again
    screen.open_delete_prompt_i18n(&catalog);
    assert!(screen.has_prompt());

    // Click Cancel button
    let cancel_btn = screen.delete_prompt.as_ref().unwrap().cancel_button.clone();
    let cancel_act = screen.handle_mouse_click_i18n(
        cancel_btn.x + 2.0,
        cancel_btn.y + 2.0,
        640.0,
        480.0,
        &catalog,
    );
    assert_eq!(cancel_act, None);
    assert!(!screen.has_prompt(), "Cancel button should dismiss modal");
}

#[test]
fn test_world_select_keyboard_controls() {
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    for i in 0..4 {
        screen.worlds.push(WorldEntry::new(
            format!("World {i}"),
            format!("dir_{i}"),
            PathBuf::from(format!("/tmp/dir_{i}")),
            i as u64,
            "Standard",
            None,
            100,
        ));
    }
    screen.selected_index = Some(0);

    // Arrow down / S
    screen.select_next();
    assert_eq!(screen.selected_index, Some(1));
    screen.select_next();
    assert_eq!(screen.selected_index, Some(2));

    // Arrow up / W
    screen.select_previous();
    assert_eq!(screen.selected_index, Some(1));

    // Open delete prompt
    assert!(screen.open_delete_prompt_i18n(&catalog));
    assert!(screen.has_prompt());

    // Cancel prompt (Escape)
    assert!(screen.cancel_prompt());
    assert!(!screen.has_prompt());

    // Reopen prompt and confirm (Enter)
    assert!(screen.open_delete_prompt_i18n(&catalog));
    let confirmed_action = screen.confirm_prompt();
    assert_eq!(
        confirmed_action,
        Some(WorldSelectAction::DeleteWorld(screen.worlds[1].clone()))
    );
    assert!(!screen.has_prompt());
}

#[test]
fn test_world_select_scrolling_and_overflow() {
    let font = BitmapFont::new_fallback(0);
    let catalog = LanguageCatalog::with_default_embedded();
    let mut screen = WorldSelectScreen::new();

    for i in 0..20 {
        screen.worlds.push(WorldEntry::new(
            format!("Overflow World {i}"),
            format!("over_{i}"),
            PathBuf::from(format!("/tmp/over_{i}")),
            i as u64,
            "Standard",
            None,
            1024,
        ));
    }

    screen.update_layout_i18n(640.0, 360.0, &catalog);
    assert!(
        screen.max_scroll > 0.0,
        "Expected max_scroll > 0 for 20 worlds"
    );

    // Scroll down with mouse wheel
    screen.handle_mouse_wheel(-2.0);
    assert!(screen.scroll_offset > 0.0);

    // Render with scrollbar
    let mut quads = Vec::new();
    screen.render(&font, 640.0, 360.0, 2, &mut quads);
    assert!(!quads.is_empty());
}
