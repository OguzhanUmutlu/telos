//! Hello Block - Telos WASM Mod
use telos_sdk::export_mod;
use telos_sdk::events::{EventFilter, ModEvent};
use telos_sdk::guest::{log, register_command, set_block, subscribe_events, LogLevel, TelosMod};

#[derive(Default)]
pub struct HelloBlock;

impl TelosMod for HelloBlock {
    fn init(&mut self) -> Result<(), String> {
        log(LogLevel::Info, "[hello_block] Mod initialized successfully!");
        subscribe_events(EventFilter::BLOCK_BROKEN.union(EventFilter::BLOCK_PLACED))?;
        register_command("glow")?;
        Ok(())
    }

    fn on_event(&mut self, event: &ModEvent) {
        match event {
            ModEvent::BlockPlaced { pos, new_state, .. } => {
                log(LogLevel::Info, &format!("[hello_block] Block placed at {pos:?} (state {new_state})"));
            }
            ModEvent::BlockBroken { pos, old_state, .. } => {
                log(LogLevel::Info, &format!("[hello_block] Block broken at {pos:?} (state {old_state})"));
            }
            _ => {}
        }
    }

    fn on_command(&mut self, cmd: &str, _args: &str) -> Result<String, String> {
        if cmd == "glow" {
            // Example: place a glowing block 2 units above origin
            set_block(0, 70, 0, 1)?;
            Ok("Glow activated!".to_string())
        } else {
            Ok(String::new())
        }
    }
}

export_mod!(HelloBlock);
