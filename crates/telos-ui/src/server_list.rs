//! Multiplayer server browser screen, LAN server discovery cards, and direct connect.

use crate::font::BitmapFont;
use crate::quad::UiQuad;
use crate::scale::snap_to_physical;

/// Information for a single server shown in the server list.
#[derive(Debug, Clone, PartialEq)]
pub struct ServerEntry {
    /// Server display name.
    pub name: String,
    /// Network address (e.g. `127.0.0.1:47679`).
    pub address: String,
    /// Server MOTD / description.
    pub motd: String,
    /// Current number of connected players.
    pub current_players: u16,
    /// Maximum player capacity.
    pub max_players: u16,
    /// Round-trip ping time in milliseconds, if measured.
    pub ping_ms: Option<u32>,
    /// Whether this server was discovered via local LAN broadcast/multicast.
    pub is_lan: bool,
}

impl ServerEntry {
    /// Creates a new server entry.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        address: impl Into<String>,
        motd: impl Into<String>,
        current_players: u16,
        max_players: u16,
        is_lan: bool,
    ) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
            motd: motd.into(),
            current_players,
            max_players,
            ping_ms: None,
            is_lan,
        }
    }
}

/// Interactive state and card layout for the multiplayer server browser.
#[derive(Debug, Clone, Default)]
pub struct ServerListScreen {
    /// Registered and discovered servers.
    pub servers: Vec<ServerEntry>,
    /// Index of selected server in `servers`, if any.
    pub selected_index: Option<usize>,
    /// Direct connect address input buffer.
    pub direct_connect_address: String,
    /// Whether the direct connect input box is active and capturing input.
    pub direct_connect_focused: bool,
    /// Vertical scroll offset in GUI pixels.
    pub scroll_offset: f32,
}

impl ServerListScreen {
    /// Creates a new empty `ServerListScreen`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or updates a LAN discovered server entry by address.
    pub fn add_or_update_lan_server(&mut self, entry: ServerEntry) {
        if let Some(existing) = self.servers.iter_mut().find(|s| s.address == entry.address) {
            existing.motd = entry.motd;
            existing.current_players = entry.current_players;
            existing.max_players = entry.max_players;
            existing.ping_ms = entry.ping_ms;
        } else {
            self.servers.push(entry);
            if self.selected_index.is_none() {
                self.selected_index = Some(0);
            }
        }
    }

    /// Selects a server card by index.
    pub fn select_server(&mut self, idx: usize) {
        if idx < self.servers.len() {
            self.selected_index = Some(idx);
        }
    }

    /// Returns a reference to the currently selected server entry, if any.
    #[must_use]
    pub fn selected_server(&self) -> Option<&ServerEntry> {
        let idx = self.selected_index?;
        self.servers.get(idx)
    }

    /// Removes LAN servers not present in `active_addresses`.
    pub fn retain_active_lan_addresses(&mut self, active_addresses: &[String]) {
        self.servers
            .retain(|s| !s.is_lan || active_addresses.contains(&s.address));
        if matches!(self.selected_index, Some(idx) if idx >= self.servers.len()) {
            self.selected_index = self.servers.len().checked_sub(1);
        }
    }
}

/// Renders the multiplayer server list screen into the output `UiQuad` buffer.
#[allow(
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss
)]
pub fn render_server_list(
    screen: &ServerListScreen,
    font: &BitmapFont,
    screen_w: u32,
    screen_h: u32,
    gui_scale: u32,
    out: &mut Vec<UiQuad>,
) {
    let scale = gui_scale.max(1);
    let screen_w_gui = screen_w as f32 / scale as f32;
    let screen_h_gui = screen_h as f32 / scale as f32;

    // 1. Full-screen backdrop
    out.push(UiQuad::solid(
        [0, 0],
        [screen_w as u16, screen_h as u16],
        UiQuad::rgba(16, 16, 16, 235),
    ));

    // 2. Header title: "Play Multiplayer"
    let title = "Play Multiplayer";
    let (title_w, _) = font.measure_text(title);
    let title_x = (screen_w_gui - title_w) / 2.0;
    font.layout_text(
        title,
        title_x,
        12.0,
        UiQuad::rgba(255, 255, 255, 255),
        true,
        scale,
        out,
    );

    // 3. Server list cards area
    let list_top_gui = 32.0f32;
    let list_bottom_gui = screen_h_gui - 52.0f32;
    let card_w_gui = 300.0f32.min(screen_w_gui - 20.0);
    let card_h_gui = 36.0f32;
    let card_x_gui = (screen_w_gui - card_w_gui) / 2.0;

    let mut current_y = list_top_gui - screen.scroll_offset;

    if screen.servers.is_empty() {
        let msg = "Scanning for LAN games...";
        let (msg_w, _) = font.measure_text(msg);
        let msg_x = (screen_w_gui - msg_w) / 2.0;
        font.layout_text(
            msg,
            msg_x,
            list_top_gui + 24.0,
            UiQuad::rgba(160, 160, 160, 255),
            true,
            scale,
            out,
        );
    } else {
        for (i, server) in screen.servers.iter().enumerate() {
            if current_y + card_h_gui < list_top_gui {
                current_y += card_h_gui + 4.0;
                continue;
            }
            if current_y > list_bottom_gui {
                break;
            }

            let is_selected = screen.selected_index == Some(i);

            let phys_x = snap_to_physical(card_x_gui, scale);
            let phys_y = snap_to_physical(current_y, scale);
            let phys_w = (card_w_gui * scale as f32).round() as u16;
            let phys_h = (card_h_gui * scale as f32).round() as u16;

            // Card background and selection border
            let bg_color = if is_selected {
                UiQuad::rgba(40, 50, 80, 220)
            } else {
                UiQuad::rgba(25, 25, 25, 200)
            };
            out.push(UiQuad::solid([phys_x, phys_y], [phys_w, phys_h], bg_color));

            if is_selected {
                out.push(UiQuad::solid(
                    [phys_x, phys_y],
                    [phys_w, scale as u16],
                    UiQuad::rgba(255, 255, 255, 255),
                ));
                out.push(UiQuad::solid(
                    [phys_x, phys_y + i32::from(phys_h) - scale.cast_signed()],
                    [phys_w, scale as u16],
                    UiQuad::rgba(255, 255, 255, 255),
                ));
            }

            // Server title + LAN badge
            let mut name_text = server.name.clone();
            if server.is_lan {
                name_text.push_str(" §a[LAN]");
            }
            font.layout_text(
                &name_text,
                card_x_gui + 4.0,
                current_y + 4.0,
                UiQuad::rgba(255, 255, 255, 255),
                true,
                scale,
                out,
            );

            // Server MOTD
            font.layout_text(
                &server.motd,
                card_x_gui + 4.0,
                current_y + 16.0,
                UiQuad::rgba(180, 180, 180, 255),
                true,
                scale,
                out,
            );

            // Player count (right aligned)
            let player_text = format!("{}/{}", server.current_players, server.max_players);
            let (pw, _) = font.measure_text(&player_text);
            font.layout_text(
                &player_text,
                card_x_gui + card_w_gui - pw - 6.0,
                current_y + 4.0,
                UiQuad::rgba(180, 180, 180, 255),
                true,
                scale,
                out,
            );

            // Ping text
            let ping_text = server
                .ping_ms
                .map_or_else(|| "---".to_string(), |p| format!("{p}ms"));
            let (ping_w, _) = font.measure_text(&ping_text);
            font.layout_text(
                &ping_text,
                card_x_gui + card_w_gui - ping_w - 6.0,
                current_y + 16.0,
                UiQuad::rgba(85, 255, 85, 255),
                true,
                scale,
                out,
            );

            current_y += card_h_gui + 4.0;
        }
    }

    // 4. Bottom action buttons
    let btn_y = screen_h_gui - 36.0;
    let btn_w = 90.0f32;
    let btn_h = 20.0f32;
    let total_btns_w = (btn_w * 3.0) + (8.0 * 2.0);
    let start_btn_x = (screen_w_gui - total_btns_w) / 2.0;

    let buttons = ["Join Server", "Direct Connect", "Cancel"];
    for (i, btn_label) in buttons.iter().enumerate() {
        let bx = start_btn_x + (i as f32 * (btn_w + 8.0));
        let phys_bx = snap_to_physical(bx, scale);
        let phys_by = snap_to_physical(btn_y, scale);
        let phys_bw = (btn_w * scale as f32).round() as u16;
        let phys_bh = (btn_h * scale as f32).round() as u16;

        let has_sel = screen.selected_index.is_some() || i > 0;
        let btn_bg = if has_sel {
            UiQuad::rgba(60, 60, 60, 255)
        } else {
            UiQuad::rgba(35, 35, 35, 255)
        };

        out.push(UiQuad::solid(
            [phys_bx, phys_by],
            [phys_bw, phys_bh],
            btn_bg,
        ));
        out.push(UiQuad::solid(
            [phys_bx, phys_by],
            [phys_bw, scale as u16],
            UiQuad::rgba(120, 120, 120, 255),
        ));

        let (lw, _) = font.measure_text(btn_label);
        let tx = bx + ((btn_w - lw) / 2.0);
        let ty = btn_y + ((btn_h - 9.0) / 2.0);
        let text_color = if has_sel {
            UiQuad::rgba(255, 255, 255, 255)
        } else {
            UiQuad::rgba(120, 120, 120, 255)
        };

        font.layout_text(btn_label, tx, ty, text_color, true, scale, out);
    }
}
