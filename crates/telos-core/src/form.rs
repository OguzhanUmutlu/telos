//! Canonical server-driven modal forms protocol models inspired by Bedrock Edition.
//!
//! Provides JSON-serializable declarations for `ActionForm` (button lists),
//! `ModalForm` (two-button confirmation dialogs), and `CustomForm` (labels, toggles,
//! sliders, dropdowns, inputs), along with client response representations.

use serde::{Deserialize, Serialize};

/// Discriminant tag indicating the image reference scheme in form buttons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FormImageType {
    /// Texture or asset path relative to the active resource pack root.
    Path,
    /// Absolute web or network URL.
    Url,
}

/// Icon image metadata attached to an `ActionForm` button.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormImage {
    /// Scheme indicating whether `data` is a resource pack path or external URL.
    #[serde(rename = "type")]
    pub image_type: FormImageType,
    /// Image URI or texture path.
    pub data: String,
}

impl FormImage {
    /// Creates a new resource pack path image reference.
    #[must_use]
    pub fn path(path: impl Into<String>) -> Self {
        Self {
            image_type: FormImageType::Path,
            data: path.into(),
        }
    }

    /// Creates a new external URL image reference.
    #[must_use]
    pub fn url(url: impl Into<String>) -> Self {
        Self {
            image_type: FormImageType::Url,
            data: url.into(),
        }
    }
}

/// Button entry inside an `ActionForm`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FormButton {
    /// Button label text.
    pub text: String,
    /// Optional icon image displayed next to the text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image: Option<FormImage>,
}

impl FormButton {
    /// Creates a simple text button without an icon.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            image: None,
        }
    }

    /// Creates a button with an attached icon image.
    #[must_use]
    pub fn with_image(text: impl Into<String>, image: FormImage) -> Self {
        Self {
            text: text.into(),
            image: Some(image),
        }
    }
}

/// Simple action button form (Bedrock `type: "form"`).
///
/// Displays a header title, descriptive content body, and a vertical list of clickable buttons.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionForm {
    /// Window title bar text.
    pub title: String,
    /// Descriptive header or markdown content text.
    pub content: String,
    /// List of action buttons.
    pub buttons: Vec<FormButton>,
}

impl ActionForm {
    /// Creates a new `ActionForm` with title and content.
    #[must_use]
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            buttons: Vec::new(),
        }
    }

    /// Appends a text button and returns `self` for fluent chaining.
    #[must_use]
    pub fn button(mut self, text: impl Into<String>) -> Self {
        self.buttons.push(FormButton::new(text));
        self
    }

    /// Appends a button with an icon image and returns `self` for fluent chaining.
    #[must_use]
    pub fn button_with_image(mut self, text: impl Into<String>, image: FormImage) -> Self {
        self.buttons.push(FormButton::with_image(text, image));
        self
    }
}

/// Two-button modal dialog (Bedrock `type: "modal"`).
///
/// Prompts the user with a confirmation query and two choice buttons (`button1` = true, `button2` = false).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModalForm {
    /// Window title bar text.
    pub title: String,
    /// Prompt content text.
    pub content: String,
    /// Affirmative button text (returns `true` on click).
    pub button1: String,
    /// Negative / cancel button text (returns `false` on click).
    pub button2: String,
}

impl ModalForm {
    /// Creates a new `ModalForm`.
    #[must_use]
    pub fn new(
        title: impl Into<String>,
        content: impl Into<String>,
        button1: impl Into<String>,
        button2: impl Into<String>,
    ) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            button1: button1.into(),
            button2: button2.into(),
        }
    }
}

/// Element type within a `CustomForm`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FormElement {
    /// Read-only informational label.
    Label {
        /// Label text content.
        text: String,
    },
    /// Boolean on/off switch toggle.
    Toggle {
        /// Setting description text.
        text: String,
        /// Initial toggle state.
        default: bool,
    },
    /// Numeric draggable slider.
    Slider {
        /// Setting description text.
        text: String,
        /// Minimum selectable value.
        min: f32,
        /// Maximum selectable value.
        max: f32,
        /// Optional step increment.
        #[serde(skip_serializing_if = "Option::is_none")]
        step: Option<f32>,
        /// Initial slider value.
        default: f32,
    },
    /// Discrete step slider with named values.
    StepSlider {
        /// Setting description text.
        text: String,
        /// Named step options.
        steps: Vec<String>,
        /// Initial selected index in `steps`.
        #[serde(default)]
        default: u32,
    },
    /// Dropdown selection menu.
    Dropdown {
        /// Setting description text.
        text: String,
        /// Selectable options.
        options: Vec<String>,
        /// Initial selected index in `options`.
        #[serde(default)]
        default: u32,
    },
    /// Single-line text input field.
    Input {
        /// Field label text.
        text: String,
        /// Placeholder text shown when input is empty.
        placeholder: String,
        /// Initial text value.
        default: String,
    },
}

/// Complex interactive form containing heterogeneous input elements (Bedrock `type: "custom_form"`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomForm {
    /// Window title bar text.
    pub title: String,
    /// Heterogeneous list of form elements.
    pub content: Vec<FormElement>,
}

impl CustomForm {
    /// Creates a new `CustomForm` with the specified title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: Vec::new(),
        }
    }

    /// Appends a read-only informational label.
    #[must_use]
    pub fn label(mut self, text: impl Into<String>) -> Self {
        self.content.push(FormElement::Label { text: text.into() });
        self
    }

    /// Appends a boolean toggle switch.
    #[must_use]
    pub fn toggle(mut self, text: impl Into<String>, default: bool) -> Self {
        self.content.push(FormElement::Toggle {
            text: text.into(),
            default,
        });
        self
    }

    /// Appends a continuous or stepped numeric slider.
    #[must_use]
    pub fn slider(
        mut self,
        text: impl Into<String>,
        min: f32,
        max: f32,
        step: Option<f32>,
        default: f32,
    ) -> Self {
        self.content.push(FormElement::Slider {
            text: text.into(),
            min,
            max,
            step,
            default: default.clamp(min, max),
        });
        self
    }

    /// Appends a discrete step slider.
    #[must_use]
    pub fn step_slider(
        mut self,
        text: impl Into<String>,
        steps: Vec<String>,
        default_index: u32,
    ) -> Self {
        self.content.push(FormElement::StepSlider {
            text: text.into(),
            steps,
            default: default_index,
        });
        self
    }

    /// Appends a dropdown options selector.
    #[must_use]
    pub fn dropdown(
        mut self,
        text: impl Into<String>,
        options: Vec<String>,
        default_index: u32,
    ) -> Self {
        self.content.push(FormElement::Dropdown {
            text: text.into(),
            options,
            default: default_index,
        });
        self
    }

    /// Appends a single-line text input field.
    #[must_use]
    pub fn input(
        mut self,
        text: impl Into<String>,
        placeholder: impl Into<String>,
        default: impl Into<String>,
    ) -> Self {
        self.content.push(FormElement::Input {
            text: text.into(),
            placeholder: placeholder.into(),
            default: default.into(),
        });
        self
    }
}

/// Unified modal form data model tagged by `"type"`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ModalFormData {
    /// Button list action form (`type: "form"`).
    #[serde(rename = "form")]
    Action(ActionForm),
    /// Two-button modal dialog (`type: "modal"`).
    #[serde(rename = "modal")]
    Modal(ModalForm),
    /// Heterogeneous interactive elements form (`type: "custom_form"`).
    #[serde(rename = "custom_form")]
    Custom(CustomForm),
}

impl ModalFormData {
    /// Serializes this form definition into a JSON string.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Deserializes a form definition from a JSON string.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if the JSON structure does not match a valid form definition.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Returns the window title text of this form.
    #[must_use]
    pub fn title(&self) -> &str {
        match self {
            Self::Action(f) => &f.title,
            Self::Modal(f) => &f.title,
            Self::Custom(f) => &f.title,
        }
    }
}

/// Heterogeneous response value from a single `CustomForm` element.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FormValue {
    /// Null representation for non-interactive elements like labels.
    Null,
    /// Boolean state from a toggle element.
    Bool(bool),
    /// Integer index from a dropdown or step slider.
    Int(i64),
    /// Floating point value from a slider.
    Float(f64),
    /// String content from a text input.
    String(String),
}

impl FormValue {
    /// Extracts a boolean value if present.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Extracts a floating point value if present.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            #[allow(clippy::cast_precision_loss)]
            Self::Int(i) => Some(*i as f64),
            _ => None,
        }
    }

    /// Extracts an integer value if present.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            #[allow(clippy::cast_possible_truncation)]
            Self::Float(f) => Some(*f as i64),
            _ => None,
        }
    }

    /// Extracts a string slice if present.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Strongly typed response payload submitted by a client in response to a modal form request.
#[derive(Debug, Clone, PartialEq)]
pub enum FormResponseData {
    /// Selected button index from an `ActionForm`.
    Action {
        /// Zero-based index of the clicked button.
        button_index: u32,
    },
    /// Boolean confirmation choice from a `ModalForm`.
    Modal {
        /// `true` if `button1` was clicked, `false` if `button2`.
        confirmed: bool,
    },
    /// Heterogeneous array of values matching each element in a `CustomForm`.
    Custom {
        /// Element response values in the same order as declared in `CustomForm::content`.
        values: Vec<FormValue>,
    },
}

impl FormResponseData {
    /// Serializes this response data into a JSON string formatted according to Bedrock protocol conventions.
    ///
    /// # Errors
    /// Returns a `serde_json::Error` on serialization failure.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        match self {
            Self::Action { button_index } => serde_json::to_string(button_index),
            Self::Modal { confirmed } => serde_json::to_string(confirmed),
            Self::Custom { values } => serde_json::to_string(values),
        }
    }

    /// Parses an `ActionForm` response JSON string (e.g. `"0"` or `"1"`).
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_action_json(json: &str) -> Result<Self, serde_json::Error> {
        let button_index: u32 = serde_json::from_str(json)?;
        Ok(Self::Action { button_index })
    }

    /// Parses a `ModalForm` response JSON string (e.g. `"true"` or `"false"`).
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_modal_json(json: &str) -> Result<Self, serde_json::Error> {
        let confirmed: bool = serde_json::from_str(json)?;
        Ok(Self::Modal { confirmed })
    }

    /// Parses a `CustomForm` response JSON string (e.g. `"[null, true, 4.0, 1, \"test\"]"`).
    ///
    /// # Errors
    /// Returns a `serde_json::Error` if parsing fails.
    pub fn from_custom_json(json: &str) -> Result<Self, serde_json::Error> {
        let values: Vec<FormValue> = serde_json::from_str(json)?;
        Ok(Self::Custom { values })
    }
}

/// Reason code specifying why a form was dismissed without a response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FormCancelReason {
    /// User closed the form (e.g. clicked close button or pressed ESC).
    UserClosed = 0,
    /// Form was dismissed because the user was busy or another window took precedence.
    UserBusy = 1,
}

impl FormCancelReason {
    /// Decodes a wire byte into a `FormCancelReason`.
    #[must_use]
    pub fn from_u8(val: u8) -> Self {
        match val {
            1 => Self::UserBusy,
            _ => Self::UserClosed,
        }
    }

    /// Returns the wire byte representation.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_form_json_roundtrip() {
        let form = ModalFormData::Action(
            ActionForm::new("Server Help", "Select a command category below:")
                .button("World Commands")
                .button_with_image(
                    "Player Controls",
                    FormImage::path("textures/items/compass.png"),
                ),
        );

        let json = form.to_json().expect("Failed to serialize ActionForm");
        assert!(json.contains("\"type\":\"form\""));
        assert!(json.contains("\"title\":\"Server Help\""));
        assert!(json.contains("\"textures/items/compass.png\""));

        let decoded = ModalFormData::from_json(&json).expect("Failed to deserialize ActionForm");
        assert_eq!(form, decoded);

        let resp = FormResponseData::Action { button_index: 1 };
        let resp_json = resp.to_json().expect("Failed to serialize response");
        assert_eq!(resp_json, "1");
        let decoded_resp = FormResponseData::from_action_json(&resp_json)
            .expect("Failed to deserialize action response");
        assert_eq!(resp, decoded_resp);
    }

    #[test]
    fn test_modal_form_json_roundtrip() {
        let form = ModalFormData::Modal(ModalForm::new(
            "Confirm Teleport",
            "Are you sure you want to teleport to spawn?",
            "Yes, Teleport",
            "Cancel",
        ));

        let json = form.to_json().expect("Failed to serialize ModalForm");
        assert!(json.contains("\"type\":\"modal\""));
        assert!(json.contains("\"button1\":\"Yes, Teleport\""));

        let decoded = ModalFormData::from_json(&json).expect("Failed to deserialize ModalForm");
        assert_eq!(form, decoded);

        let resp = FormResponseData::Modal { confirmed: true };
        let resp_json = resp.to_json().expect("Failed to serialize response");
        assert_eq!(resp_json, "true");
        let decoded_resp = FormResponseData::from_modal_json(&resp_json)
            .expect("Failed to deserialize modal response");
        assert_eq!(resp, decoded_resp);
    }

    #[test]
    fn test_custom_form_json_roundtrip() {
        let form = ModalFormData::Custom(
            CustomForm::new("Player Settings")
                .label("Configure your client preferences")
                .toggle("Auto-Jump", true)
                .slider("Render Distance", 2.0, 32.0, Some(1.0), 8.0)
                .dropdown(
                    "Theme",
                    vec!["Classic".to_string(), "Modern".to_string()],
                    1,
                )
                .input("Nickname", "Enter username...", "Steve"),
        );

        let json = form.to_json().expect("Failed to serialize CustomForm");
        assert!(json.contains("\"type\":\"custom_form\""));
        assert!(json.contains("\"Auto-Jump\""));

        let decoded = ModalFormData::from_json(&json).expect("Failed to deserialize CustomForm");
        assert_eq!(form, decoded);

        let resp = FormResponseData::Custom {
            values: vec![
                FormValue::Null,
                FormValue::Bool(false),
                FormValue::Float(16.5),
                FormValue::Int(0),
                FormValue::String("Alex".to_string()),
            ],
        };
        let resp_json = resp.to_json().expect("Failed to serialize custom response");
        let decoded_resp = FormResponseData::from_custom_json(&resp_json)
            .expect("Failed to deserialize custom response");
        assert_eq!(resp, decoded_resp);
    }
}
