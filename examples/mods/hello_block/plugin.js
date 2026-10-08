// Telos QuickJS Server Plugin — Hello Block
// Demonstrates chat filtering, block break protection, and periodic tick hooks.

console.log("[hello_block] QuickJS plugin initialized");

// Chat hook: inspects or transforms player chat messages
function onPlayerChat(player, message) {
    if (message.startsWith("!hello")) {
        console.log(`[hello_block] Player ${player} said hello!`);
        return `[HelloBlock] Greetings, ${player}! Welcome to Telos.`;
    }
    return message;
}

// Block break hook: authorizes or cancels block breaking
function onBlockBreak(ctx) {
    // ctx: { playerId, blockId, x, y, z }
    // Example: prevent breaking at deep subterranean bedrock levels
    if (ctx.y < -1000) {
        console.log(`[hello_block] Protected deep world at Y=${ctx.y}`);
        return false;
    }
    return true;
}

// Periodic tick hook (ticked 20 times per second)
let tickCount = 0;
function onTick(currentTick) {
    tickCount++;
    if (tickCount % 600 === 0) { // Every 30 seconds
        console.log(`[hello_block] Server tick heartbeat: ${currentTick}`);
    }
}
