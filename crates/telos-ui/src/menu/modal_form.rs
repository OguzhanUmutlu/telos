//! Bedrock-style server-driven modal forms screen and widget renderer.
//!
//! Renders declarative forms sent by the server (`ActionForm`, `ModalForm`, `CustomForm`)
//! into responsive pure-CPU `UiQuad` primitives, supporting scrolling, button selection,
//! toggles, sliders, dropdowns, text inputs, and keyboard navigation.

use telos_core::form::{FormCancelReason, FormElement, FormResponseData, FormValue, ModalFormData};

use crate::font::BitmapFont;
use crate::menu::widgets::{MenuButton, MenuSlider, MenuTextInput};
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;

/// Action emitted by user interaction with a modal form.
#[derive(Debug, Clone, PartialEq)]
pub enum ModalFormAction {
    /// Form was submitted with response payload.
    Submit(FormResponseData),
    /// Form was closed or cancelled.
    Cancel(FormCancelReason),
}

/// Dynamic widget state for an element in a `CustomForm`.
#[derive(Debug, Clone, PartialEq)]
pub enum CustomWidgetState {
    /// Read-only informational label.
    Label {
        /// Text content to display.
        text: String,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Height of label element in GUI pixels.
        height: f32,
    },
    /// Boolean on/off toggle switch.
    Toggle {
        /// Setting label.
        text: String,
        /// Current boolean state.
        current: bool,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Interactive toggle button.
        button: MenuButton,
    },
    /// Continuous or stepped numeric slider.
    Slider {
        /// Setting label.
        text: String,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Optional step increment.
        step: Option<f32>,
        /// Interactive slider widget.
        slider: MenuSlider,
    },
    /// Discrete step slider with named steps.
    StepSlider {
        /// Setting label.
        text: String,
        /// Named step options.
        steps: Vec<String>,
        /// Currently active step index.
        current: usize,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Step previous button.
        prev_btn: MenuButton,
        /// Step next button.
        next_btn: MenuButton,
    },
    /// Dropdown selector menu.
    Dropdown {
        /// Setting label.
        text: String,
        /// Selectable options.
        options: Vec<String>,
        /// Currently active option index.
        current: usize,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Option previous button.
        prev_btn: MenuButton,
        /// Option next button.
        next_btn: MenuButton,
    },
    /// Single-line text input field.
    Input {
        /// Field label text.
        text: String,
        /// Base Y position relative to content origin.
        base_y: f32,
        /// Interactive text input widget.
        input: MenuTextInput,
    },
}

/// Retained modal form screen rendering responsive dialogs on top of gameplay.
#[derive(Debug, Clone)]
pub struct ModalFormScreen {
    /// Server-assigned unique form identifier.
    pub form_id: u32,
    /// Canonical form declaration data.
    pub data: ModalFormData,
    /// Top-left X in GUI pixels.
    pub dialog_x: f32,
    /// Top-left Y in GUI pixels.
    pub dialog_y: f32,
    /// Width of dialog in GUI pixels.
    pub dialog_w: f32,
    /// Height of dialog in GUI pixels.
    pub dialog_h: f32,
    /// Title bar close button [X].
    pub close_btn: MenuButton,
    /// Current vertical scroll offset in GUI pixels.
    pub scroll_offset: f32,
    /// Maximum scrollable distance in GUI pixels.
    pub max_scroll: f32,
    /// Total content height before scrolling.
    pub total_content_height: f32,
    /// Action buttons for `ActionForm`.
    pub action_buttons: Vec<MenuButton>,
    /// Button 1 (Confirm) for `ModalForm`.
    pub modal_btn1: MenuButton,
    /// Button 2 (Cancel) for `ModalForm`.
    pub modal_btn2: MenuButton,
    /// Interactive element states for `CustomForm`.
    pub custom_elements: Vec<CustomWidgetState>,
    /// Submit button for `CustomForm`.
    pub custom_submit_btn: MenuButton,
    /// Cancel button for `CustomForm`.
    pub custom_cancel_btn: MenuButton,
    /// Currently dragging slider index within `custom_elements`.
    pub active_drag_slider: Option<usize>,
}

impl ModalFormScreen {
    /// Creates a new `ModalFormScreen` for the specified form ID and declaration data.
    #[allow(clippy::too_many_lines)]
    #[must_use]
    pub fn new(form_id: u32, data: ModalFormData) -> Self {
        let action_buttons = if let ModalFormData::Action(ref action) = data {
            action
                .buttons
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    let label = if b.image.is_some() {
                        format!("[*] {}", b.text)
                    } else {
                        b.text.clone()
                    };
                    MenuButton::new(i as u32, 0.0, 0.0, 100.0, 22.0, label)
                })
                .collect()
        } else {
            Vec::new()
        };

        let (modal_btn1, modal_btn2) = if let ModalFormData::Modal(ref modal) = data {
            (
                MenuButton::new(1, 0.0, 0.0, 100.0, 24.0, modal.button1.clone()),
                MenuButton::new(2, 0.0, 0.0, 100.0, 24.0, modal.button2.clone()),
            )
        } else {
            (
                MenuButton::new(1, 0.0, 0.0, 100.0, 24.0, "Confirm"),
                MenuButton::new(2, 0.0, 0.0, 100.0, 24.0, "Cancel"),
            )
        };

        let mut custom_elements = Vec::new();
        if let ModalFormData::Custom(ref custom) = data {
            for (i, elem) in custom.content.iter().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                let elem_id = i as u32;
                match elem {
                    FormElement::Label { text } => {
                        custom_elements.push(CustomWidgetState::Label {
                            text: text.clone(),
                            base_y: 0.0,
                            height: 16.0,
                        });
                    }
                    FormElement::Toggle { text, default } => {
                        let btn_label = if *default { "ON" } else { "OFF" };
                        let btn = MenuButton::new(1000 + elem_id, 0.0, 0.0, 60.0, 20.0, btn_label);
                        custom_elements.push(CustomWidgetState::Toggle {
                            text: text.clone(),
                            current: *default,
                            base_y: 0.0,
                            button: btn,
                        });
                    }
                    FormElement::Slider {
                        text,
                        min,
                        max,
                        step,
                        default,
                    } => {
                        let is_int = step.is_some_and(|s| (s - 1.0).abs() < 1e-4);
                        let slider = MenuSlider::new(
                            2000 + elem_id,
                            0.0,
                            0.0,
                            200.0,
                            20.0,
                            text.clone(),
                            *min,
                            *max,
                            *default,
                            "",
                            is_int,
                        );
                        custom_elements.push(CustomWidgetState::Slider {
                            text: text.clone(),
                            base_y: 0.0,
                            step: *step,
                            slider,
                        });
                    }
                    FormElement::StepSlider {
                        text,
                        steps,
                        default,
                    } => {
                        let current = (*default as usize).min(steps.len().saturating_sub(1));
                        let prev_btn =
                            MenuButton::new(3000 + elem_id * 2, 0.0, 0.0, 24.0, 20.0, "<");
                        let next_btn =
                            MenuButton::new(3000 + elem_id * 2 + 1, 0.0, 0.0, 24.0, 20.0, ">");
                        custom_elements.push(CustomWidgetState::StepSlider {
                            text: text.clone(),
                            steps: steps.clone(),
                            current,
                            base_y: 0.0,
                            prev_btn,
                            next_btn,
                        });
                    }
                    FormElement::Dropdown {
                        text,
                        options,
                        default,
                    } => {
                        let current = (*default as usize).min(options.len().saturating_sub(1));
                        let prev_btn =
                            MenuButton::new(4000 + elem_id * 2, 0.0, 0.0, 24.0, 20.0, "<");
                        let next_btn =
                            MenuButton::new(4000 + elem_id * 2 + 1, 0.0, 0.0, 24.0, 20.0, ">");
                        custom_elements.push(CustomWidgetState::Dropdown {
                            text: text.clone(),
                            options: options.clone(),
                            current,
                            base_y: 0.0,
                            prev_btn,
                            next_btn,
                        });
                    }
                    FormElement::Input {
                        text,
                        placeholder,
                        default,
                    } => {
                        let mut input = MenuTextInput::new(
                            5000 + elem_id,
                            0.0,
                            0.0,
                            200.0,
                            20.0,
                            placeholder.clone(),
                        );
                        input.text.clone_from(default);
                        input.cursor_pos = input.text.len();
                        custom_elements.push(CustomWidgetState::Input {
                            text: text.clone(),
                            base_y: 0.0,
                            input,
                        });
                    }
                }
            }
        }

        Self {
            form_id,
            data,
            dialog_x: 0.0,
            dialog_y: 0.0,
            dialog_w: 380.0,
            dialog_h: 240.0,
            close_btn: MenuButton::new(999, 0.0, 0.0, 18.0, 18.0, "x"),
            scroll_offset: 0.0,
            max_scroll: 0.0,
            total_content_height: 0.0,
            action_buttons,
            modal_btn1,
            modal_btn2,
            custom_elements,
            custom_submit_btn: MenuButton::new(888, 0.0, 0.0, 120.0, 24.0, "Submit"),
            custom_cancel_btn: MenuButton::new(889, 0.0, 0.0, 120.0, 24.0, "Cancel"),
            active_drag_slider: None,
        }
    }

    /// Recomputes layout geometry and positions all child widgets for current GUI dimensions.
    #[allow(clippy::too_many_lines)]
    pub fn update_layout(&mut self, width_gui: f32, height_gui: f32) {
        let title_bar_h = 24.0;
        let pad_x = 14.0;
        let pad_top = 8.0;

        self.dialog_w = (width_gui - 32.0).clamp(320.0, 420.0);
        let inner_w = self.dialog_w - pad_x * 2.0;

        match &self.data {
            ModalFormData::Action(action) => {
                let content_lines = action.content.lines().count().max(1);
                #[allow(clippy::cast_precision_loss)]
                let text_h = content_lines as f32 * 10.0 + 8.0;
                let btn_h = 22.0;
                let btn_gap = 4.0;
                #[allow(clippy::cast_precision_loss)]
                let buttons_total_h = self.action_buttons.len() as f32 * (btn_h + btn_gap) + 12.0;

                self.total_content_height = text_h + buttons_total_h;
                let ideal_dialog_h = title_bar_h + pad_top + self.total_content_height + 12.0;
                let max_avail_h = (height_gui - 36.0).max(180.0);

                self.dialog_h = ideal_dialog_h.min(max_avail_h);
                self.dialog_x = ((width_gui - self.dialog_w) * 0.5).round();
                self.dialog_y = ((height_gui - self.dialog_h) * 0.5).round();

                let visible_content_h = self.dialog_h - title_bar_h - pad_top - 8.0;
                self.max_scroll = (self.total_content_height - visible_content_h).max(0.0);
                self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll);

                self.close_btn.x = self.dialog_x + self.dialog_w - 22.0;
                self.close_btn.y = self.dialog_y + 3.0;

                let start_btn_y =
                    self.dialog_y + title_bar_h + pad_top + text_h - self.scroll_offset;
                for (i, btn) in self.action_buttons.iter_mut().enumerate() {
                    btn.x = self.dialog_x + pad_x;
                    #[allow(clippy::cast_precision_loss)]
                    {
                        btn.y = start_btn_y + i as f32 * (btn_h + btn_gap);
                    }
                    btn.width = inner_w;
                    btn.height = btn_h;
                }
            }
            ModalFormData::Modal(modal) => {
                let content_lines = modal.content.lines().count().max(1);
                #[allow(clippy::cast_precision_loss)]
                let text_h = content_lines as f32 * 10.0 + 12.0;
                let bottom_bar_h = 36.0;

                self.total_content_height = text_h;
                let ideal_dialog_h = title_bar_h + pad_top + text_h + bottom_bar_h + 12.0;
                let max_avail_h = (height_gui - 36.0).max(180.0);

                self.dialog_h = ideal_dialog_h.min(max_avail_h);
                self.dialog_x = ((width_gui - self.dialog_w) * 0.5).round();
                self.dialog_y = ((height_gui - self.dialog_h) * 0.5).round();

                self.max_scroll = 0.0;
                self.scroll_offset = 0.0;

                self.close_btn.x = self.dialog_x + self.dialog_w - 22.0;
                self.close_btn.y = self.dialog_y + 3.0;

                let btn_w = (inner_w - 10.0) * 0.5;
                let btn_y = self.dialog_y + self.dialog_h - 32.0;
                self.modal_btn1.x = self.dialog_x + pad_x;
                self.modal_btn1.y = btn_y;
                self.modal_btn1.width = btn_w;
                self.modal_btn1.height = 24.0;

                self.modal_btn2.x = self.dialog_x + pad_x + btn_w + 10.0;
                self.modal_btn2.y = btn_y;
                self.modal_btn2.width = btn_w;
                self.modal_btn2.height = 24.0;
            }
            ModalFormData::Custom(_) => {
                let bottom_bar_h = 36.0;
                let mut current_elem_y = 0.0f32;

                for elem in &mut self.custom_elements {
                    match elem {
                        CustomWidgetState::Label { base_y, height, .. } => {
                            *base_y = current_elem_y;
                            current_elem_y += *height + 4.0;
                        }
                        CustomWidgetState::Toggle { base_y, .. } => {
                            *base_y = current_elem_y;
                            current_elem_y += 24.0 + 6.0;
                        }
                        CustomWidgetState::Slider { base_y, .. }
                        | CustomWidgetState::StepSlider { base_y, .. }
                        | CustomWidgetState::Dropdown { base_y, .. }
                        | CustomWidgetState::Input { base_y, .. } => {
                            *base_y = current_elem_y;
                            current_elem_y += 34.0 + 6.0;
                        }
                    }
                }

                self.total_content_height = current_elem_y;
                let ideal_dialog_h = title_bar_h + pad_top + current_elem_y + bottom_bar_h + 12.0;
                let max_avail_h = (height_gui - 36.0).max(220.0);

                self.dialog_h = ideal_dialog_h.min(max_avail_h);
                self.dialog_x = ((width_gui - self.dialog_w) * 0.5).round();
                self.dialog_y = ((height_gui - self.dialog_h) * 0.5).round();

                let visible_content_h = self.dialog_h - title_bar_h - pad_top - bottom_bar_h;
                self.max_scroll = (self.total_content_height - visible_content_h).max(0.0);
                self.scroll_offset = self.scroll_offset.clamp(0.0, self.max_scroll);

                self.close_btn.x = self.dialog_x + self.dialog_w - 22.0;
                self.close_btn.y = self.dialog_y + 3.0;

                let content_origin_y = self.dialog_y + title_bar_h + pad_top - self.scroll_offset;

                for elem in &mut self.custom_elements {
                    match elem {
                        CustomWidgetState::Label { .. } => {}
                        CustomWidgetState::Toggle { base_y, button, .. } => {
                            button.x = self.dialog_x + self.dialog_w - pad_x - 70.0;
                            button.y = content_origin_y + *base_y;
                            button.width = 70.0;
                            button.height = 20.0;
                        }
                        CustomWidgetState::Slider { base_y, slider, .. } => {
                            slider.x = self.dialog_x + pad_x;
                            slider.y = content_origin_y + *base_y + 12.0;
                            slider.width = inner_w;
                            slider.height = 20.0;
                        }
                        CustomWidgetState::StepSlider {
                            base_y,
                            prev_btn,
                            next_btn,
                            ..
                        }
                        | CustomWidgetState::Dropdown {
                            base_y,
                            prev_btn,
                            next_btn,
                            ..
                        } => {
                            let elem_y = content_origin_y + *base_y + 12.0;
                            prev_btn.x = self.dialog_x + pad_x;
                            prev_btn.y = elem_y;
                            prev_btn.width = 24.0;
                            prev_btn.height = 20.0;

                            next_btn.x = self.dialog_x + self.dialog_w - pad_x - 24.0;
                            next_btn.y = elem_y;
                            next_btn.width = 24.0;
                            next_btn.height = 20.0;
                        }
                        CustomWidgetState::Input { base_y, input, .. } => {
                            input.x = self.dialog_x + pad_x;
                            input.y = content_origin_y + *base_y + 12.0;
                            input.width = inner_w;
                            input.height = 20.0;
                        }
                    }
                }

                let btn_w = (inner_w - 10.0) * 0.5;
                let btn_y = self.dialog_y + self.dialog_h - 32.0;
                self.custom_submit_btn.x = self.dialog_x + pad_x;
                self.custom_submit_btn.y = btn_y;
                self.custom_submit_btn.width = btn_w;
                self.custom_submit_btn.height = 24.0;

                self.custom_cancel_btn.x = self.dialog_x + pad_x + btn_w + 10.0;
                self.custom_cancel_btn.y = btn_y;
                self.custom_cancel_btn.width = btn_w;
                self.custom_cancel_btn.height = 24.0;
            }
        }
    }

    /// Handles mouse motion in GUI units and updates hover states.
    pub fn handle_mouse_move(&mut self, mouse_x: f32, mouse_y: f32) {
        self.close_btn.hovered = self.close_btn.contains(mouse_x, mouse_y);

        if let Some(idx) = self.active_drag_slider {
            if let Some(CustomWidgetState::Slider { step, slider, .. }) =
                self.custom_elements.get_mut(idx)
            {
                slider.update_from_mouse_x(mouse_x);
                if let Some(s) = step
                    && *s > 0.0
                {
                    slider.value = (slider.value / *s).round() * *s;
                    slider.value = slider.value.clamp(slider.min, slider.max);
                }
            }
            return;
        }

        let content_min_y = self.dialog_y + 24.0;
        let content_max_y = match &self.data {
            ModalFormData::Action(_) => self.dialog_y + self.dialog_h,
            ModalFormData::Modal(_) | ModalFormData::Custom(_) => {
                self.dialog_y + self.dialog_h - 36.0
            }
        };

        match &mut self.data {
            ModalFormData::Action(_) => {
                for btn in &mut self.action_buttons {
                    let in_view = btn.y + btn.height >= content_min_y && btn.y <= content_max_y;
                    btn.hovered = in_view && btn.contains(mouse_x, mouse_y);
                }
            }
            ModalFormData::Modal(_) => {
                self.modal_btn1.hovered = self.modal_btn1.contains(mouse_x, mouse_y);
                self.modal_btn2.hovered = self.modal_btn2.contains(mouse_x, mouse_y);
            }
            ModalFormData::Custom(_) => {
                for elem in &mut self.custom_elements {
                    match elem {
                        CustomWidgetState::Label { .. } => {}
                        CustomWidgetState::Toggle { button, .. } => {
                            let in_view = button.y + button.height >= content_min_y
                                && button.y <= content_max_y;
                            button.hovered = in_view && button.contains(mouse_x, mouse_y);
                        }
                        CustomWidgetState::Slider { slider, .. } => {
                            let in_view = slider.y + slider.height >= content_min_y
                                && slider.y <= content_max_y;
                            slider.hovered = in_view && slider.contains(mouse_x, mouse_y);
                        }
                        CustomWidgetState::StepSlider {
                            prev_btn, next_btn, ..
                        }
                        | CustomWidgetState::Dropdown {
                            prev_btn, next_btn, ..
                        } => {
                            let in_view = prev_btn.y + prev_btn.height >= content_min_y
                                && prev_btn.y <= content_max_y;
                            prev_btn.hovered = in_view && prev_btn.contains(mouse_x, mouse_y);
                            next_btn.hovered = in_view && next_btn.contains(mouse_x, mouse_y);
                        }
                        CustomWidgetState::Input { input, .. } => {
                            let in_view =
                                input.y + input.height >= content_min_y && input.y <= content_max_y;
                            input.hovered = in_view && input.contains(mouse_x, mouse_y);
                        }
                    }
                }
                self.custom_submit_btn.hovered = self.custom_submit_btn.contains(mouse_x, mouse_y);
                self.custom_cancel_btn.hovered = self.custom_cancel_btn.contains(mouse_x, mouse_y);
            }
        }
    }

    /// Handles mouse button click down in GUI units.
    #[allow(clippy::too_many_lines)]
    pub fn handle_mouse_click(&mut self, mouse_x: f32, mouse_y: f32) -> Option<ModalFormAction> {
        if self.close_btn.contains(mouse_x, mouse_y) {
            return Some(ModalFormAction::Cancel(FormCancelReason::UserClosed));
        }

        let content_min_y = self.dialog_y + 24.0;
        let content_max_y = match &self.data {
            ModalFormData::Action(_) => self.dialog_y + self.dialog_h,
            ModalFormData::Modal(_) | ModalFormData::Custom(_) => {
                self.dialog_y + self.dialog_h - 36.0
            }
        };

        match &mut self.data {
            ModalFormData::Action(_) => {
                for (i, btn) in self.action_buttons.iter().enumerate() {
                    let in_view = btn.y + btn.height >= content_min_y && btn.y <= content_max_y;
                    if in_view && btn.contains(mouse_x, mouse_y) {
                        #[allow(clippy::cast_possible_truncation)]
                        return Some(ModalFormAction::Submit(FormResponseData::Action {
                            button_index: i as u32,
                        }));
                    }
                }
            }
            ModalFormData::Modal(_) => {
                if self.modal_btn1.contains(mouse_x, mouse_y) {
                    return Some(ModalFormAction::Submit(FormResponseData::Modal {
                        confirmed: true,
                    }));
                }
                if self.modal_btn2.contains(mouse_x, mouse_y) {
                    return Some(ModalFormAction::Submit(FormResponseData::Modal {
                        confirmed: false,
                    }));
                }
            }
            ModalFormData::Custom(_) => {
                if self.custom_submit_btn.contains(mouse_x, mouse_y) {
                    let mut values = Vec::with_capacity(self.custom_elements.len());
                    for elem in &self.custom_elements {
                        match elem {
                            CustomWidgetState::Label { .. } => values.push(FormValue::Null),
                            CustomWidgetState::Toggle { current, .. } => {
                                values.push(FormValue::Bool(*current));
                            }
                            CustomWidgetState::Slider { slider, .. } =>
                            {
                                #[allow(clippy::cast_possible_truncation)]
                                if slider.is_int {
                                    values.push(FormValue::Int(slider.value.round() as i64));
                                } else {
                                    values.push(FormValue::Float(f64::from(slider.value)));
                                }
                            }
                            CustomWidgetState::StepSlider { current, .. }
                            | CustomWidgetState::Dropdown { current, .. } => {
                                #[allow(clippy::cast_possible_wrap)]
                                values.push(FormValue::Int(*current as i64));
                            }
                            CustomWidgetState::Input { input, .. } => {
                                values.push(FormValue::String(input.text.clone()));
                            }
                        }
                    }
                    return Some(ModalFormAction::Submit(FormResponseData::Custom { values }));
                }
                if self.custom_cancel_btn.contains(mouse_x, mouse_y) {
                    return Some(ModalFormAction::Cancel(FormCancelReason::UserClosed));
                }

                for (idx, elem) in self.custom_elements.iter_mut().enumerate() {
                    match elem {
                        CustomWidgetState::Label { .. } => {}
                        CustomWidgetState::Toggle {
                            current, button, ..
                        } => {
                            let in_view = button.y + button.height >= content_min_y
                                && button.y <= content_max_y;
                            if in_view && button.contains(mouse_x, mouse_y) {
                                *current = !*current;
                                button.label = if *current {
                                    "ON".to_string()
                                } else {
                                    "OFF".to_string()
                                };
                            }
                        }
                        CustomWidgetState::Slider { step, slider, .. } => {
                            let in_view = slider.y + slider.height >= content_min_y
                                && slider.y <= content_max_y;
                            if in_view && slider.contains(mouse_x, mouse_y) {
                                self.active_drag_slider = Some(idx);
                                slider.dragging = true;
                                slider.update_from_mouse_x(mouse_x);
                                if let Some(s) = step
                                    && *s > 0.0
                                {
                                    slider.value = (slider.value / *s).round() * *s;
                                    slider.value = slider.value.clamp(slider.min, slider.max);
                                }
                            }
                        }
                        CustomWidgetState::StepSlider {
                            steps,
                            current,
                            prev_btn,
                            next_btn,
                            ..
                        } => {
                            let in_view = prev_btn.y + prev_btn.height >= content_min_y
                                && prev_btn.y <= content_max_y;
                            if in_view {
                                if prev_btn.contains(mouse_x, mouse_y) && *current > 0 {
                                    *current -= 1;
                                } else if next_btn.contains(mouse_x, mouse_y)
                                    && *current + 1 < steps.len()
                                {
                                    *current += 1;
                                }
                            }
                        }
                        CustomWidgetState::Dropdown {
                            options,
                            current,
                            prev_btn,
                            next_btn,
                            ..
                        } => {
                            let in_view = prev_btn.y + prev_btn.height >= content_min_y
                                && prev_btn.y <= content_max_y;
                            if in_view {
                                if prev_btn.contains(mouse_x, mouse_y) && *current > 0 {
                                    *current -= 1;
                                } else if next_btn.contains(mouse_x, mouse_y)
                                    && *current + 1 < options.len()
                                {
                                    *current += 1;
                                }
                            }
                        }
                        CustomWidgetState::Input { input, .. } => {
                            let in_view =
                                input.y + input.height >= content_min_y && input.y <= content_max_y;
                            input.focused = in_view && input.contains(mouse_x, mouse_y);
                        }
                    }
                }
            }
        }

        None
    }

    /// Handles mouse button release to cancel active slider dragging.
    pub fn handle_mouse_up(&mut self) {
        if let Some(idx) = self.active_drag_slider.take()
            && let Some(CustomWidgetState::Slider { slider, .. }) =
                self.custom_elements.get_mut(idx)
        {
            slider.dragging = false;
        }
    }

    /// Handles scroll wheel delta to scroll dialog content.
    pub fn handle_scroll(&mut self, delta_y: f32) {
        if self.max_scroll > 0.0 {
            self.scroll_offset = (self.scroll_offset - delta_y * 18.0).clamp(0.0, self.max_scroll);
        }
    }

    /// Handles character typing for focused text input fields in `CustomForm`.
    pub fn handle_char(&mut self, ch: char) {
        for elem in &mut self.custom_elements {
            if let CustomWidgetState::Input { input, .. } = elem
                && input.focused
            {
                input.insert_char(ch);
            }
        }
    }

    /// Handles backspace key for focused text input fields in `CustomForm`.
    pub fn handle_backspace(&mut self) {
        for elem in &mut self.custom_elements {
            if let CustomWidgetState::Input { input, .. } = elem
                && input.focused
            {
                input.backspace();
            }
        }
    }

    /// Checks if mouse cursor is currently hovering over any interactive child widget.
    #[must_use]
    pub fn is_hovered(&self) -> bool {
        if self.close_btn.hovered {
            return true;
        }
        match &self.data {
            ModalFormData::Action(_) => self.action_buttons.iter().any(|b| b.hovered),
            ModalFormData::Modal(_) => self.modal_btn1.hovered || self.modal_btn2.hovered,
            ModalFormData::Custom(_) => {
                self.custom_submit_btn.hovered
                    || self.custom_cancel_btn.hovered
                    || self.custom_elements.iter().any(|elem| match elem {
                        CustomWidgetState::Label { .. } => false,
                        CustomWidgetState::Toggle { button, .. } => button.hovered,
                        CustomWidgetState::Slider { slider, .. } => {
                            slider.hovered || slider.dragging
                        }
                        CustomWidgetState::StepSlider {
                            prev_btn, next_btn, ..
                        }
                        | CustomWidgetState::Dropdown {
                            prev_btn, next_btn, ..
                        } => prev_btn.hovered || next_btn.hovered,
                        CustomWidgetState::Input { input, .. } => input.hovered,
                    })
            }
        }
    }

    /// Renders the entire modal form dialog and its child widgets into the UI quad buffer.
    #[allow(
        clippy::too_many_lines,
        clippy::cast_possible_wrap,
        clippy::cast_possible_truncation,
        clippy::similar_names
    )]
    pub fn render(&self, font: &BitmapFont, scale: u32, tick: u64, out: &mut Vec<UiQuad>) {
        let border_th = scale.max(1) as u16;
        let px_x = snap_to_physical(self.dialog_x, scale);
        let px_y = snap_to_physical(self.dialog_y, scale);
        let px_w = snap_to_physical(self.dialog_w, scale) as u16;
        let px_h = snap_to_physical(self.dialog_h, scale) as u16;
        let title_px_h = snap_to_physical(24.0, scale) as u16;

        // Dialog background panel
        out.push(UiQuad::solid(
            [px_x, px_y],
            [px_w, px_h],
            UiQuad::rgba(25, 25, 30, 248),
        ));

        // Dialog outer borders
        let border_col = UiQuad::rgba(65, 65, 75, 255);
        out.push(UiQuad::solid([px_x, px_y], [px_w, border_th], border_col));
        out.push(UiQuad::solid(
            [px_x, px_y + i32::from(px_h) - i32::from(border_th)],
            [px_w, border_th],
            border_col,
        ));
        out.push(UiQuad::solid([px_x, px_y], [border_th, px_h], border_col));
        out.push(UiQuad::solid(
            [px_x + i32::from(px_w) - i32::from(border_th), px_y],
            [border_th, px_h],
            border_col,
        ));

        // Title bar header background
        out.push(UiQuad::solid(
            [px_x + i32::from(border_th), px_y + i32::from(border_th)],
            [px_w.saturating_sub(border_th * 2), title_px_h],
            UiQuad::rgba(35, 35, 42, 255),
        ));

        // Gold accent bottom line under title
        let gold_col = UiQuad::rgba(220, 180, 50, 255);
        let accent_th = scale.max(1) as u16;
        out.push(UiQuad::solid(
            [
                px_x + i32::from(border_th),
                px_y + i32::from(border_th) + i32::from(title_px_h) - i32::from(accent_th),
            ],
            [px_w.saturating_sub(border_th * 2), accent_th],
            gold_col,
        ));

        // Title text
        let title_str = self.data.title();
        let title_tx = self.dialog_x + 12.0;
        let title_ty = self.dialog_y + 7.0;
        font.layout_text(
            title_str,
            title_tx,
            title_ty,
            UiQuad::rgba(255, 255, 255, 255),
            true,
            scale,
            out,
        );

        // Close button [X]
        self.close_btn.render(font, scale, out);

        let content_min_y = self.dialog_y + 24.0;
        let content_max_y = match &self.data {
            ModalFormData::Action(_) => self.dialog_y + self.dialog_h,
            ModalFormData::Modal(_) | ModalFormData::Custom(_) => {
                self.dialog_y + self.dialog_h - 36.0
            }
        };

        match &self.data {
            ModalFormData::Action(action) => {
                let mut text_y = self.dialog_y + 24.0 + 8.0 - self.scroll_offset;
                for line in action.content.lines() {
                    if text_y >= content_min_y && text_y + 8.0 <= content_max_y {
                        font.layout_text(
                            line,
                            self.dialog_x + 14.0,
                            text_y,
                            UiQuad::rgba(220, 220, 220, 255),
                            true,
                            scale,
                            out,
                        );
                    }
                    text_y += 10.0;
                }

                for btn in &self.action_buttons {
                    if btn.y + btn.height >= content_min_y && btn.y <= content_max_y {
                        btn.render(font, scale, out);
                    }
                }
            }
            ModalFormData::Modal(modal) => {
                let mut text_y = self.dialog_y + 24.0 + 10.0;
                for line in modal.content.lines() {
                    if text_y >= content_min_y && text_y + 8.0 <= content_max_y {
                        font.layout_text(
                            line,
                            self.dialog_x + 14.0,
                            text_y,
                            UiQuad::rgba(220, 220, 220, 255),
                            true,
                            scale,
                            out,
                        );
                    }
                    text_y += 10.0;
                }

                self.modal_btn1.render(font, scale, out);
                self.modal_btn2.render(font, scale, out);
            }
            ModalFormData::Custom(_) => {
                for elem in &self.custom_elements {
                    match elem {
                        CustomWidgetState::Label {
                            text,
                            base_y,
                            height: _,
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly,
                                    UiQuad::rgba(220, 220, 220, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                        }
                        CustomWidgetState::Toggle {
                            text,
                            current: _,
                            base_y,
                            button,
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly + 4.0,
                                    UiQuad::rgba(240, 240, 240, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                            if button.y + button.height >= content_min_y
                                && button.y <= content_max_y
                            {
                                button.render(font, scale, out);
                            }
                        }
                        CustomWidgetState::Slider {
                            text,
                            base_y,
                            slider,
                            ..
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly,
                                    UiQuad::rgba(200, 200, 210, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                            if slider.y + slider.height >= content_min_y
                                && slider.y <= content_max_y
                            {
                                slider.render(font, scale, out);
                            }
                        }
                        CustomWidgetState::StepSlider {
                            text,
                            steps,
                            current,
                            base_y,
                            prev_btn,
                            next_btn,
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly,
                                    UiQuad::rgba(200, 200, 210, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                            if prev_btn.y + prev_btn.height >= content_min_y
                                && prev_btn.y <= content_max_y
                            {
                                prev_btn.render(font, scale, out);
                                next_btn.render(font, scale, out);

                                let val_str = steps.get(*current).map_or("", String::as_str);
                                let (tw, _) = font.measure_text(val_str);
                                let box_x = prev_btn.x + prev_btn.width + 4.0;
                                let box_w = (next_btn.x - 4.0) - box_x;
                                let tx = box_x + (box_w - tw) * 0.5;
                                font.layout_text(
                                    val_str,
                                    tx,
                                    prev_btn.y + 6.0,
                                    UiQuad::rgba(255, 255, 200, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                        }
                        CustomWidgetState::Dropdown {
                            text,
                            options,
                            current,
                            base_y,
                            prev_btn,
                            next_btn,
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly,
                                    UiQuad::rgba(200, 200, 210, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                            if prev_btn.y + prev_btn.height >= content_min_y
                                && prev_btn.y <= content_max_y
                            {
                                prev_btn.render(font, scale, out);
                                next_btn.render(font, scale, out);

                                let val_str = options.get(*current).map_or("", String::as_str);
                                let (tw, _) = font.measure_text(val_str);
                                let box_x = prev_btn.x + prev_btn.width + 4.0;
                                let box_w = (next_btn.x - 4.0) - box_x;
                                let tx = box_x + (box_w - tw) * 0.5;
                                font.layout_text(
                                    val_str,
                                    tx,
                                    prev_btn.y + 6.0,
                                    UiQuad::rgba(255, 255, 200, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                        }
                        CustomWidgetState::Input {
                            text,
                            base_y,
                            input,
                        } => {
                            let ly = self.dialog_y + 24.0 + 8.0 + *base_y - self.scroll_offset;
                            if ly >= content_min_y && ly + 8.0 <= content_max_y {
                                font.layout_text(
                                    text,
                                    self.dialog_x + 14.0,
                                    ly,
                                    UiQuad::rgba(200, 200, 210, 255),
                                    true,
                                    scale,
                                    out,
                                );
                            }
                            if input.y + input.height >= content_min_y && input.y <= content_max_y {
                                input.render(font, scale, tick, out);
                            }
                        }
                    }
                }

                self.custom_submit_btn.render(font, scale, out);
                self.custom_cancel_btn.render(font, scale, out);
            }
        }

        // Scrollbar indicator
        if self.max_scroll > 0.0 {
            let track_x = snap_to_physical(self.dialog_x + self.dialog_w - 7.0, scale);
            let track_y = snap_to_physical(content_min_y + 2.0, scale);
            let track_w = snap_to_physical(4.0, scale) as u16;
            let track_h = snap_to_physical(content_max_y - content_min_y - 4.0, scale) as u16;

            out.push(UiQuad::solid(
                [track_x, track_y],
                [track_w, track_h],
                UiQuad::rgba(40, 40, 45, 180),
            ));

            let visible_h = content_max_y - content_min_y;
            let thumb_h_gui =
                (visible_h * (visible_h / self.total_content_height)).clamp(14.0, visible_h);
            let thumb_y_gui = content_min_y
                + 2.0
                + (visible_h - thumb_h_gui) * (self.scroll_offset / self.max_scroll);
            let thumb_y = snap_to_physical(thumb_y_gui, scale);
            let thumb_h = snap_to_physical(thumb_h_gui, scale) as u16;

            out.push(UiQuad::solid(
                [track_x, thumb_y],
                [track_w, thumb_h],
                UiQuad::rgba(160, 160, 175, 220),
            ));
        }
    }
}
