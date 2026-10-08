/**
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
