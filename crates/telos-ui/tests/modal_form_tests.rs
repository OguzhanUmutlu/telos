//! Tests for Bedrock-style modal form screen and widget interactions.

#![allow(clippy::float_cmp)]

use telos_core::form::{
    ActionForm, CustomForm, FormCancelReason, FormResponseData, FormValue, ModalForm, ModalFormData,
};
use telos_ui::font::BitmapFont;
use telos_ui::menu::modal_form::{CustomWidgetState, ModalFormAction, ModalFormScreen};

#[test]
fn test_action_form_layout_and_click() {
    let form = ActionForm::new("Server Teleport Menu", "Select destination:")
        .button("Spawn Point")
        .button("Mining World")
        .button("Player Hub");
    let mut screen = ModalFormScreen::new(101, ModalFormData::Action(form));
    screen.update_layout(800.0, 600.0);

    assert_eq!(screen.form_id, 101);
    assert_eq!(screen.action_buttons.len(), 3);
    assert!(screen.dialog_w >= 320.0);
    assert!(screen.dialog_h >= 100.0);

    // Hit test first button
    let btn0 = &screen.action_buttons[0];
    let action = screen.handle_mouse_click(btn0.x + 5.0, btn0.y + 5.0);
    assert_eq!(
        action,
        Some(ModalFormAction::Submit(FormResponseData::Action {
            button_index: 0
        }))
    );

    // Hit test third button
    let btn2 = &screen.action_buttons[2];
    let action = screen.handle_mouse_click(btn2.x + 5.0, btn2.y + 5.0);
    assert_eq!(
        action,
        Some(ModalFormAction::Submit(FormResponseData::Action {
            button_index: 2
        }))
    );

    // Hit test close button [X]
    let action = screen.handle_mouse_click(screen.close_btn.x + 2.0, screen.close_btn.y + 2.0);
    assert_eq!(
        action,
        Some(ModalFormAction::Cancel(FormCancelReason::UserClosed))
    );
}

#[test]
fn test_modal_form_confirm_and_cancel() {
    let form = ModalForm::new(
        "Confirm Warp",
        "Teleporting to wilderness costs 10 coins. Proceed?",
        "Yes, Warp!",
        "Nevermind",
    );
    let mut screen = ModalFormScreen::new(102, ModalFormData::Modal(form));
    screen.update_layout(800.0, 600.0);

    // Hit test button 1 (Confirm)
    let action = screen.handle_mouse_click(screen.modal_btn1.x + 5.0, screen.modal_btn1.y + 5.0);
    assert_eq!(
        action,
        Some(ModalFormAction::Submit(FormResponseData::Modal {
            confirmed: true
        }))
    );

    // Hit test button 2 (Cancel)
    let action = screen.handle_mouse_click(screen.modal_btn2.x + 5.0, screen.modal_btn2.y + 5.0);
    assert_eq!(
        action,
        Some(ModalFormAction::Submit(FormResponseData::Modal {
            confirmed: false
        }))
    );
}

#[test]
fn test_custom_form_heterogeneous_elements_and_submit() {
    let form = CustomForm::new("Server Settings")
        .label("Configure your player preferences below:")
        .toggle("PvP Protection", true)
        .slider("View Distance", 4.0, 32.0, Some(1.0), 16.0)
        .step_slider(
            "Difficulty",
            vec![
                "Peaceful".to_string(),
                "Easy".to_string(),
                "Normal".to_string(),
                "Hard".to_string(),
            ],
            2,
        )
        .dropdown(
            "Language",
            vec![
                "English".to_string(),
                "German".to_string(),
                "Turkish".to_string(),
            ],
            0,
        )
        .input("Nickname", "Enter nickname...", "Player1");

    let mut screen = ModalFormScreen::new(103, ModalFormData::Custom(form));
    screen.update_layout(800.0, 600.0);

    assert_eq!(screen.custom_elements.len(), 6);

    // 1. Toggle switch flip: click toggle button
    if let CustomWidgetState::Toggle {
        ref button,
        current,
        ..
    } = screen.custom_elements[1]
    {
        assert!(current);
        let click_x = button.x + 5.0;
        let click_y = button.y + 5.0;
        screen.handle_mouse_click(click_x, click_y);
    }
    if let CustomWidgetState::Toggle { current, .. } = screen.custom_elements[1] {
        assert!(!current); // flipped to false
    }

    // 2. StepSlider cycle next: click next_btn
    if let CustomWidgetState::StepSlider {
        ref next_btn,
        current,
        ..
    } = screen.custom_elements[3]
    {
        assert_eq!(current, 2);
        let click_x = next_btn.x + 5.0;
        let click_y = next_btn.y + 5.0;
        screen.handle_mouse_click(click_x, click_y);
    }
    if let CustomWidgetState::StepSlider { current, .. } = screen.custom_elements[3] {
        assert_eq!(current, 3); // incremented to 3 ("Hard")
    }

    // 3. Dropdown cycle next: click next_btn
    if let CustomWidgetState::Dropdown {
        ref next_btn,
        current,
        ..
    } = screen.custom_elements[4]
    {
        assert_eq!(current, 0);
        let click_x = next_btn.x + 5.0;
        let click_y = next_btn.y + 5.0;
        screen.handle_mouse_click(click_x, click_y);
    }
    if let CustomWidgetState::Dropdown { current, .. } = screen.custom_elements[4] {
        assert_eq!(current, 1); // incremented to 1 ("German")
    }

    // 4. Text input typing
    if let CustomWidgetState::Input { ref input, .. } = screen.custom_elements[5] {
        let click_x = input.x + 5.0;
        let click_y = input.y + 5.0;
        screen.handle_mouse_click(click_x, click_y);
    }
    screen.handle_backspace();
    screen.handle_char('2');
    if let CustomWidgetState::Input { ref input, .. } = screen.custom_elements[5] {
        assert_eq!(input.text, "Player2");
    }

    // 5. Submit form
    let submit_action = screen.handle_mouse_click(
        screen.custom_submit_btn.x + 5.0,
        screen.custom_submit_btn.y + 5.0,
    );

    match submit_action {
        Some(ModalFormAction::Submit(FormResponseData::Custom { values })) => {
            assert_eq!(values.len(), 6);
            assert_eq!(values[0], FormValue::Null); // Label
            assert_eq!(values[1], FormValue::Bool(false)); // Toggle flipped
            assert_eq!(values[2], FormValue::Int(16)); // Slider is_int: 16
            assert_eq!(values[3], FormValue::Int(3)); // StepSlider ("Hard")
            assert_eq!(values[4], FormValue::Int(1)); // Dropdown ("German")
            assert_eq!(values[5], FormValue::String("Player2".to_string())); // Nickname
        }
        other => panic!("Expected custom form submission, got {other:?}"),
    }
}

#[test]
fn test_modal_form_scrolling_and_rendering() {
    let mut form = ActionForm::new("Long Help Index", "Commands:\nLine 1\nLine 2\nLine 3");
    for i in 0..20 {
        form = form.button(format!("Command #{i}"));
    }

    let mut screen = ModalFormScreen::new(104, ModalFormData::Action(form));
    // Small viewport forcing scroll
    screen.update_layout(640.0, 300.0);

    assert!(screen.max_scroll > 0.0);
    assert_eq!(screen.scroll_offset, 0.0);

    // Scroll down
    screen.handle_scroll(-2.0);
    assert!(screen.scroll_offset > 0.0);

    // Test rendering produces valid quads
    let font = BitmapFont::new_fallback(0);

    let mut quads = Vec::new();
    screen.render(&font, 2, 60, &mut quads);
    assert!(!quads.is_empty());
}
