//! JavaScript and TypeScript definitions and template generators for `QuickJS` plugins.

/// Generates TypeScript declaration file content (`telos.d.ts`) for Telos JavaScript plugins.
#[must_use]
pub fn generate_typescript_declarations() -> &'static str {
    r"/**
 * Telos Modding API TypeScript Declarations
 * For QuickJS server plugins and resource pack scripts.
 */

declare interface BlockBreakContext {
    /** Entity net ID of the player performing the break. */
    playerId: number;
    /** Numeric block state ID of the broken block. */
    blockId: number;
    /** X coordinate in world space. */
    x: number;
    /** Y coordinate in world space. */
    y: number;
    /** Z coordinate in world space. */
    z: number;
}

/**
 * Called when a player sends a chat message.
 * Return false or null to cancel/suppress the message.
 * Return a string to replace the message content.
 * Return true or undefined to allow the message unchanged.
 */
declare function onPlayerChat(player: string, message: string): string | boolean | null | undefined;

/**
 * Called when a player attempts to break a block.
 * Return false to cancel the break (e.g. region protection).
 * Return true or undefined to allow the break.
 */
declare function onBlockBreak(context: BlockBreakContext): boolean | undefined;

/**
 * Called on every server tick (20 times per second).
 */
declare function onTick(tickCount: number): void;
"
}

/// Generates a starter template JavaScript plugin.
#[must_use]
pub fn generate_starter_plugin_js(mod_name: &str) -> String {
    format!(
        r#"// Telos Server Plugin — {mod_name}
// Executed in sandboxed QuickJS runtime with deterministic fuel budgeting.

console.log("[{mod_name}] Plugin initialized");

// Filter or format player chat messages
function onPlayerChat(player, message) {{
    // Example: add a custom prefix or filter words
    if (message.startsWith("!ping")) {{
        console.log(`[${{player}}] Ping!`);
        return `[Mod] Pong, ${{player}}!`;
    }}
    return message;
}}

// Block break authorization hook
function onBlockBreak(ctx) {{
    // ctx: {{ playerId, blockId, x, y, z }}
    // Return false to prevent breaking protected blocks
    if (ctx.y < -500) {{
        console.log(`Protected bedrock region at Y=${{ctx.y}}`);
        return false;
    }}
    return true;
}}

// Periodic tick hook (20 TPS)
let tickCounter = 0;
function onTick(tick) {{
    tickCounter++;
    if (tickCounter % 1200 === 0) {{ // Every 60 seconds
        console.log(`[{mod_name}] 1 minute tick heartbeat: ${{tick}}`);
    }}
}}
"#
    )
}
