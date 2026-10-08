//! Entrypoint export macros for Telos guest modules.

/// Macro to export unmangled entrypoints for a type implementing `TelosMod`.
///
/// Generates:
/// - `telos_init`
/// - `_start` (when compiling for `wasm32`)
/// - `telos_on_event`
/// - `telos_on_command`
#[macro_export]
macro_rules! export_mod {
    ($mod_type:ty) => {
        static mut MOD_INSTANCE: Option<$mod_type> = None;

        /// Host initialization entrypoint.
        #[unsafe(no_mangle)]
        pub extern "C" fn telos_init() {
            // SAFETY: Single-threaded WASM runtime initialization.
            unsafe {
                let mut instance: $mod_type = ::core::default::Default::default();
                let _ = $crate::guest::TelosMod::init(&mut instance);
                MOD_INSTANCE = Some(instance);
            }
        }

        /// WASI entrypoint alias.
        #[cfg(target_arch = "wasm32")]
        #[unsafe(no_mangle)]
        pub extern "C" fn _start() {
            telos_init();
        }

        /// Host event dispatch entrypoint.
        #[unsafe(no_mangle)]
        pub extern "C" fn telos_on_event(event_id: i32, p1: i64, p2: i64, p3: i64, p4: i64) -> i32 {
            // SAFETY: In single-threaded guest store, instance is initialized or created.
            unsafe {
                if let Some(ref mut instance) = MOD_INSTANCE {
                    if let Some(event) =
                        $crate::events::ModEvent::from_abi(event_id, p1, p2, p3, p4)
                    {
                        $crate::guest::TelosMod::on_event(instance, &event);
                    }
                }
            }
            0
        }

        /// Host command execution entrypoint.
        #[unsafe(no_mangle)]
        pub extern "C" fn telos_on_command(_cmd_hash: u32, _arg_len: i64) -> i32 {
            // SAFETY: In single-threaded guest store.
            unsafe {
                if let Some(ref mut instance) = MOD_INSTANCE {
                    let _ = $crate::guest::TelosMod::on_command(instance, "", "");
                }
            }
            0
        }
    };
}
