//! Standard built-in engine commands (/help, /time, /weather, /tp, /give, /spawnmob, /kill, /clear, /say).

use crate::command::tree::{ArgumentType, CommandDispatcher, CommandNode, CommandOutput};

/// Registers all standard built-in commands into the dispatcher.
pub fn register_builtins(dispatcher: &mut CommandDispatcher) {
    register_help(dispatcher);
    register_time(dispatcher);
    register_weather(dispatcher);
    register_tp(dispatcher);
    register_give(dispatcher);
    register_spawnmob(dispatcher);
    register_kill(dispatcher);
    register_clear(dispatcher);
    register_say(dispatcher);
}

fn register_help(dispatcher: &mut CommandDispatcher) {
    let help_node = CommandNode::literal("help")
        .with_tooltip("Shows list of available commands or help on a specific command")
        .executes(|_| {
            CommandOutput::success(
                "Available commands: /help, /time, /weather, /tp, /give, /spawnmob, /kill, /clear, /say"
            )
        })
        .then(
            CommandNode::argument("command", ArgumentType::Word)
                .with_tooltip("Specific command name")
                .executes(|ctx| {
                    let cmd = ctx.get_string("command").unwrap_or("");
                    match cmd {
                        "time" => CommandOutput::success("/time <set|query> <day|night|noon|midnight|<ticks>>"),
                        "weather" => CommandOutput::success("/weather <clear|rain|thunder> [duration]"),
                        "tp" => CommandOutput::success("/tp [target] <x y z>"),
                        "give" => CommandOutput::success("/give <target> <item> [count]"),
                        "spawnmob" => CommandOutput::success("/spawnmob <mob> [x y z]"),
                        "kill" => CommandOutput::success("/kill [target]"),
                        "clear" => CommandOutput::success("/clear [target]"),
                        "say" => CommandOutput::success("/say <message>"),
                        _ => CommandOutput::failure(format!("Unknown command: '{cmd}'")),
                    }
                }),
        );

    dispatcher.register(help_node);
}

fn register_time(dispatcher: &mut CommandDispatcher) {
    let time_node = CommandNode::literal("time")
        .with_tooltip("Changes or queries the world game time")
        .then(
            CommandNode::literal("query")
                .with_tooltip("Query current world time")
                .executes(|_| CommandOutput::success("Querying world time")),
        )
        .then(
            CommandNode::literal("set")
                .with_tooltip("Set the world time")
                .then(
                    CommandNode::literal("day")
                        .with_tooltip("Set time to morning (1000 ticks)")
                        .executes(|_| CommandOutput::success("Set the time to 1000")),
                )
                .then(
                    CommandNode::literal("noon")
                        .with_tooltip("Set time to midday (6000 ticks)")
                        .executes(|_| CommandOutput::success("Set the time to 6000")),
                )
                .then(
                    CommandNode::literal("night")
                        .with_tooltip("Set time to nightfall (13000 ticks)")
                        .executes(|_| CommandOutput::success("Set the time to 13000")),
                )
                .then(
                    CommandNode::literal("midnight")
                        .with_tooltip("Set time to midnight (18000 ticks)")
                        .executes(|_| CommandOutput::success("Set the time to 18000")),
                )
                .then(
                    CommandNode::argument(
                        "ticks",
                        ArgumentType::Integer {
                            min: Some(0),
                            max: Some(24000),
                        },
                    )
                    .with_tooltip("Time in ticks [0..24000]")
                    .executes(|ctx| {
                        let ticks = ctx.get_int("ticks").unwrap_or(0);
                        CommandOutput::success(format!("Set the time to {ticks}"))
                    }),
                ),
        );

    dispatcher.register(time_node);
}

fn register_weather(dispatcher: &mut CommandDispatcher) {
    let weather_node = CommandNode::literal("weather")
        .with_tooltip("Sets the world weather")
        .then(
            CommandNode::literal("clear")
                .with_tooltip("Clear sky and sunshine")
                .executes(|_| CommandOutput::success("Set weather to clear")),
        )
        .then(
            CommandNode::literal("rain")
                .with_tooltip("Overcast sky with rain or snowfall")
                .executes(|_| CommandOutput::success("Set weather to rain")),
        )
        .then(
            CommandNode::literal("thunder")
                .with_tooltip("Thunderstorm with lightning strikes")
                .executes(|_| CommandOutput::success("Set weather to thunder")),
        );

    dispatcher.register(weather_node);
}

fn register_tp(dispatcher: &mut CommandDispatcher) {
    let tp_node = CommandNode::literal("tp")
        .with_tooltip("Teleports entities (players, mobs)")
        .then(
            CommandNode::argument("destination", ArgumentType::Vec3)
                .with_tooltip("Destination coordinates <x y z>")
                .executes(|ctx| {
                    if let Some(pos_arg) = ctx.get_vec3("destination") {
                        let target_pos = pos_arg.resolve(ctx.executor_pos, ctx.executor_rot);
                        CommandOutput::success(format!(
                            "Teleported {} to {:.2}, {:.2}, {:.2}",
                            ctx.executor_name, target_pos.x, target_pos.y, target_pos.z
                        ))
                    } else {
                        CommandOutput::failure("Invalid coordinates")
                    }
                }),
        )
        .then(
            CommandNode::argument("target", ArgumentType::Entity)
                .with_tooltip("Entity to teleport")
                .then(
                    CommandNode::argument("destination", ArgumentType::Vec3)
                        .with_tooltip("Destination coordinates <x y z>")
                        .executes(|ctx| {
                            let target = ctx.get_selector("target");
                            let target_desc = target.map_or("target", |s| match &s.selector_type {
                                crate::command::selector::SelectorType::Named(name) => {
                                    name.as_str()
                                }
                                _ => "@target",
                            });
                            if let Some(pos_arg) = ctx.get_vec3("destination") {
                                let target_pos =
                                    pos_arg.resolve(ctx.executor_pos, ctx.executor_rot);
                                CommandOutput::success(format!(
                                    "Teleported {target_desc} to {:.2}, {:.2}, {:.2}",
                                    target_pos.x, target_pos.y, target_pos.z
                                ))
                            } else {
                                CommandOutput::failure("Invalid coordinates")
                            }
                        }),
                ),
        );

    dispatcher.register(tp_node);
}

fn register_give(dispatcher: &mut CommandDispatcher) {
    let give_node = CommandNode::literal("give")
        .with_tooltip("Gives an item to one or more players")
        .then(
            CommandNode::argument("target", ArgumentType::Entity)
                .with_tooltip("Target player(s)")
                .then(
                    CommandNode::argument("item", ArgumentType::Identifier)
                        .with_tooltip("Item identifier (e.g. telos:diamond)")
                        .executes(|ctx| {
                            let item = ctx.get_string("item").unwrap_or("telos:air");
                            CommandOutput::success(format!("Gave 1 [{item}]"))
                        })
                        .then(
                            CommandNode::argument(
                                "count",
                                ArgumentType::Integer {
                                    min: Some(1),
                                    max: Some(64),
                                },
                            )
                            .with_tooltip("Quantity to give (1..64)")
                            .executes(|ctx| {
                                let item = ctx.get_string("item").unwrap_or("telos:air");
                                let count = ctx.get_int("count").unwrap_or(1);
                                CommandOutput::success(format!("Gave {count} [{item}]"))
                            }),
                        ),
                ),
        );

    dispatcher.register(give_node);
}

fn register_spawnmob(dispatcher: &mut CommandDispatcher) {
    let spawnmob_node = CommandNode::literal("spawnmob")
        .with_tooltip("Spawns a mob entity in the world")
        .then(
            CommandNode::argument("mob", ArgumentType::Word)
                .with_tooltip("Mob type (zombie, cow, pig)")
                .executes(|ctx| {
                    let mob = ctx.get_string("mob").unwrap_or("zombie");
                    let p = ctx.executor_pos;
                    CommandOutput::success(format!(
                        "Spawned mob {mob} at {:.1}, {:.1}, {:.1}",
                        p.x, p.y, p.z
                    ))
                })
                .then(
                    CommandNode::argument("pos", ArgumentType::Vec3)
                        .with_tooltip("Spawn coordinates <x y z>")
                        .executes(|ctx| {
                            let mob = ctx.get_string("mob").unwrap_or("zombie");
                            if let Some(pos_arg) = ctx.get_vec3("pos") {
                                let target_pos =
                                    pos_arg.resolve(ctx.executor_pos, ctx.executor_rot);
                                CommandOutput::success(format!(
                                    "Spawned mob {mob} at {:.1}, {:.1}, {:.1}",
                                    target_pos.x, target_pos.y, target_pos.z
                                ))
                            } else {
                                CommandOutput::failure("Invalid coordinates")
                            }
                        }),
                ),
        );

    dispatcher.register(spawnmob_node);
}

fn register_kill(dispatcher: &mut CommandDispatcher) {
    let kill_node = CommandNode::literal("kill")
        .with_tooltip("Kills entities (players, mobs)")
        .executes(|ctx| CommandOutput::success(format!("Killed {}", ctx.executor_name)))
        .then(
            CommandNode::argument("target", ArgumentType::Entity)
                .with_tooltip("Target entity to kill")
                .executes(|_| CommandOutput::success("Killed target entity")),
        );

    dispatcher.register(kill_node);
}

fn register_clear(dispatcher: &mut CommandDispatcher) {
    let clear_node = CommandNode::literal("clear")
        .with_tooltip("Clears items from player inventory")
        .executes(|ctx| {
            CommandOutput::success(format!("Cleared the inventory of {}", ctx.executor_name))
        })
        .then(
            CommandNode::argument("target", ArgumentType::Entity)
                .with_tooltip("Target player to clear")
                .executes(|_| CommandOutput::success("Cleared the inventory of target")),
        );

    dispatcher.register(clear_node);
}

fn register_say(dispatcher: &mut CommandDispatcher) {
    let say_node = CommandNode::literal("say")
        .with_tooltip("Broadcasts a message to all players")
        .then(
            CommandNode::argument("message", ArgumentType::GreedyString)
                .with_tooltip("Message text to broadcast")
                .executes(|ctx| {
                    let msg = ctx.get_string("message").unwrap_or("");
                    CommandOutput::success(format!("[{}] {msg}", ctx.executor_name))
                }),
        );

    dispatcher.register(say_node);
}
