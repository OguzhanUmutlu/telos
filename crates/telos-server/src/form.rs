//! Server-side form manager state, command catalog, and response dispatch handlers.

use telos_core::form::{ActionForm, ModalFormData};

/// Metadata entry describing a command in the interactive help browser.
#[derive(Debug, Clone, Copy)]
pub struct HelpCommandEntry {
    /// Command keyword without leading slash.
    pub name: &'static str,
    /// Concise single-sentence purpose summary.
    pub summary: &'static str,
    /// Formal command syntax pattern.
    pub syntax: &'static str,
    /// Detailed usage examples and descriptions.
    pub examples: &'static str,
    /// Default example command string to execute on button click.
    pub default_example: &'static str,
}

/// Catalog of all built-in server commands available in the interactive `/help` form.
pub static HELP_COMMANDS: &[HelpCommandEntry] = &[
    HelpCommandEntry {
        name: "time",
        summary: "Changes or queries the world game time and day/night cycle.",
        syntax: "/time <set|query> <day|noon|night|midnight|<ticks>>",
        examples: "  /time set day       - Set to morning (1,000 ticks)\n  /time set noon      - Set to midday (6,000 ticks)\n  /time set night     - Set to nightfall (13,000 ticks)\n  /time query         - Display current tick time",
        default_example: "/time query",
    },
    HelpCommandEntry {
        name: "weather",
        summary: "Controls atmospheric precipitation (clear, rain, thunderstorms).",
        syntax: "/weather <clear|rain|thunder> [duration_ticks]",
        examples: "  /weather clear      - Clear sunny skies\n  /weather rain       - Overcast and rainfall\n  /weather thunder    - Thunderstorm with lightning",
        default_example: "/weather clear",
    },
    HelpCommandEntry {
        name: "tp",
        summary: "Teleports the player or entity to exact or relative coordinates.",
        syntax: "/tp [target] <x y z>",
        examples: "  /tp 0 65 0          - Teleport to coordinate origin\n  /tp ~ ~10 ~         - Leap 10 blocks upward",
        default_example: "/tp ~ ~5 ~",
    },
    HelpCommandEntry {
        name: "give",
        summary: "Gives items to player inventory.",
        syntax: "/give <target> <item> [count]",
        examples: "  /give self iron_pickaxe 1\n  /give self bread 16\n  /give self cobblestone 64",
        default_example: "/give self bread 16",
    },
    HelpCommandEntry {
        name: "spawnmob",
        summary: "Spawns mobs into the world at current or target position.",
        syntax: "/spawnmob <zombie|cow|pig> [x y z]",
        examples: "  /spawnmob zombie    - Spawn hostile zombie\n  /spawnmob cow       - Spawn passive cow\n  /spawnmob pig       - Spawn passive pig",
        default_example: "/spawnmob cow",
    },
    HelpCommandEntry {
        name: "effect",
        summary: "Applies or clears status effects and potion modifiers.",
        syntax: "/effect <give|clear> [target] [effect] [seconds] [amplifier]",
        examples: "  /effect give self speed 30 1    - Speed II for 30s\n  /effect give self jump_boost 30 1\n  /effect clear                   - Remove all status effects",
        default_example: "/effect give self speed 30 1",
    },
    HelpCommandEntry {
        name: "enchant",
        summary: "Applies enchantments to the player's held item or armor.",
        syntax: "/enchant <target> <enchantment> [level]",
        examples: "  /enchant self sharpness 3       - Sharpness III\n  /enchant self efficiency 5      - Efficiency V\n  /enchant self unbreaking 3      - Unbreaking III",
        default_example: "/enchant self unbreaking 3",
    },
    HelpCommandEntry {
        name: "kill",
        summary: "Inflicts lethal void damage to player or target entities.",
        syntax: "/kill [target]",
        examples: "  /kill self          - Respawn player\n  /kill @e[type=zombie]",
        default_example: "/kill self",
    },
    HelpCommandEntry {
        name: "clear",
        summary: "Removes all items from player inventory.",
        syntax: "/clear [target]",
        examples: "  /clear              - Clear entire player inventory\n  /clear self",
        default_example: "/clear",
    },
    HelpCommandEntry {
        name: "world",
        summary: "Lists loaded world dimensions or teleports between them.",
        syntax: "/world <list|tp <world_name> [x y z]>",
        examples: "  /world list         - Display all active dimensions\n  /world tp overworld 0 70 0",
        default_example: "/world list",
    },
    HelpCommandEntry {
        name: "say",
        summary: "Broadcasts an announcement message to all connected players.",
        syntax: "/say <message>",
        examples: "  /say Welcome to Telos server!\n  /say Server restarting in 5 minutes",
        default_example: "/say Hello from Telos!",
    },
    HelpCommandEntry {
        name: "help",
        summary: "Displays command help index and interactive syntax browser.",
        syntax: "/help [command|page]",
        examples: "  /help               - Open interactive form\n  /help time          - Inspect /time syntax",
        default_example: "/help",
    },
];

/// Number of command buttons rendered per page in the `/help` `ActionForm`.
pub const HELP_PAGE_SIZE: usize = 6;

/// Builds an `ActionForm` for a specific page of the command index.
#[must_use]
pub fn build_help_index_form(page: usize) -> (ModalFormData, usize) {
    let total_pages = HELP_COMMANDS.len().div_ceil(HELP_PAGE_SIZE);
    let current_page = page.min(total_pages.saturating_sub(1));
    let start_idx = current_page * HELP_PAGE_SIZE;
    let end_idx = (start_idx + HELP_PAGE_SIZE).min(HELP_COMMANDS.len());

    let mut form = ActionForm::new(
        format!("Command Index (Page {}/{total_pages})", current_page + 1),
        "Click any command below to inspect syntax, parameters, and examples:",
    );

    for cmd in &HELP_COMMANDS[start_idx..end_idx] {
        form = form.button(format!("/{} - {}", cmd.name, cmd.summary));
    }

    if current_page + 1 < total_pages {
        form = form.button(format!(
            "[ Next Page (Page {}/{total_pages}) > ]",
            current_page + 2
        ));
    }
    if current_page > 0 {
        form = form.button(format!(
            "[ < Previous Page (Page {current_page}/{total_pages}) ]"
        ));
    }
    form = form.button("[ Close ]");

    (ModalFormData::Action(form), current_page)
}

/// Builds an `ActionForm` displaying syntax and usage examples for a specific command.
#[must_use]
pub fn build_help_detail_form(command_name: &str) -> (ModalFormData, Option<&'static str>) {
    let clean_name = command_name.trim_start_matches('/').to_lowercase();
    let entry = HELP_COMMANDS.iter().find(|c| c.name == clean_name);
    let (title, content, default_example) = match entry {
        Some(cmd) => (
            format!("/{} Command Reference", cmd.name),
            format!(
                "Description:\n  {}\n\nSyntax:\n  {}\n\nExamples:\n{}",
                cmd.summary, cmd.syntax, cmd.examples
            ),
            Some(cmd.default_example),
        ),
        None => (
            format!("Unknown Command: /{clean_name}"),
            format!("The command '/{clean_name}' is not in the built-in command catalog."),
            None,
        ),
    };

    let mut form = ActionForm::new(title, content).button("[ < Back to Command Index ]");
    if let Some(example) = default_example {
        form = form.button(format!("Execute: {example}"));
    }
    form = form.button("[ Close ]");

    (ModalFormData::Action(form), default_example)
}

/// Dispatched form interaction handler kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormHandlerKind {
    /// Paginated `/help` index browsing registered commands.
    HelpIndex {
        /// Active zero-based page number.
        page: usize,
    },
    /// `/help <command>` details view with back button and example execution.
    HelpCommandDetail {
        /// Inspected command name.
        command_name: String,
        /// Source page index to return to on back button.
        return_page: usize,
        /// Example command to execute if run button is clicked.
        example_command: Option<String>,
    },
    /// Custom action form button callback.
    CustomAction {
        /// User-defined tag for action routing.
        tag: String,
    },
}

/// Context for an outstanding server-driven form sent to a client.
#[derive(Debug, Clone)]
pub struct PendingForm {
    /// Tick timestamp when the form request was sent.
    pub sent_tick: u64,
    /// Handler tag or data to route the response.
    pub handler: FormHandlerKind,
}
