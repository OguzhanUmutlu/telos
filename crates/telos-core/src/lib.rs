//! # telos-core
//!
//! Core primitive types, coordinate systems, identifiers, and telemetry for the voxel engine.

pub mod coords;
pub mod dirs;
pub mod form;
pub mod i18n;
pub mod ident;
pub mod raycast;
pub mod telemetry;
pub mod time;

pub use form::{
    ActionForm, CustomForm, FormButton, FormCancelReason, FormElement, FormImage, FormImageType,
    FormResponseData, FormValue, ModalForm, ModalFormData,
};

pub use i18n::{
    I18nError, LanguageCatalog, Text, TextArg, detect_system_locale, format_pattern,
    language_display_name, parse_posix_locale,
};

pub use coords::{
    BlockPos, CHUNK_EDGE, CHUNK_MASK, CHUNK_SHIFT, CHUNK_VOLUME, ChunkPos, Face, LocalPos,
    REGION_EDGE_CHUNKS, REGION_MASK, REGION_SHIFT, REGION_VOLUME_CHUNKS, RegionPos,
};
pub use dirs::{APP_NAME, AppDirs};
pub use ident::{DEFAULT_NAMESPACE, Identifier, ParseIdentError};
pub use raycast::{RaycastHit, raycast_voxels};
pub use telemetry::{TelemetryConfig, init_telemetry};
pub use time::{
    DAY_TICKS, FixedTimestep, LUNAR_CYCLE_DAYS, MIDNIGHT_TICKS, NOON_TICKS, SUNRISE_TICKS,
    SUNSET_TICKS, daylight_factor, moon_direction, moon_phase, sun_angle, sun_direction,
    sunset_factor,
};
