//! Sandboxed JavaScript runtime and execution controller over `QuickJS`.
//!
//! Enforces memory quotas, max stack size, execution timeouts, and instruction fuel caps.
//! Zero ambient filesystem, process, or network access is permitted.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rquickjs::{Context, Function, Object, Runtime};
use std::sync::Mutex;

use crate::error::{ModError, ModResult};

/// Configuration parameters for the sandboxed JavaScript runtime.
#[derive(Debug, Clone)]
pub struct JsSandboxConfig {
    /// Maximum heap memory allocation in bytes (default: 16 MiB).
    pub memory_limit: usize,
    /// Maximum call stack depth size in bytes (default: 512 KiB).
    pub max_stack_size: usize,
    /// Maximum wall-clock execution duration for a single script invocation (default: 5 ms).
    pub timeout_duration: Duration,
    /// Maximum instruction/step limit for a single script invocation (default: 100,000 steps).
    pub max_instructions: u64,
}

impl Default for JsSandboxConfig {
    fn default() -> Self {
        Self {
            memory_limit: 16 * 1024 * 1024, // 16 MiB
            max_stack_size: 512 * 1024,     // 512 KiB
            timeout_duration: Duration::from_millis(5),
            max_instructions: 100_000,
        }
    }
}

/// Controller managing execution timeout and instruction limit interrupts.
#[derive(Clone)]
pub struct ExecutionController {
    deadline: Arc<Mutex<Option<Instant>>>,
    instruction_count: Arc<AtomicU64>,
    max_instructions: Arc<AtomicU64>,
    interrupted: Arc<AtomicBool>,
}

impl ExecutionController {
    /// Creates a new execution controller.
    #[must_use]
    pub fn new() -> Self {
        Self {
            deadline: Arc::new(Mutex::new(None)),
            instruction_count: Arc::new(AtomicU64::new(0)),
            max_instructions: Arc::new(AtomicU64::new(0)),
            interrupted: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Resets the controller for a new execution invocation.
    pub fn begin(&self, timeout: Duration, max_ops: u64) {
        let deadline = Instant::now() + timeout;
        if let Ok(mut g) = self.deadline.lock() {
            *g = Some(deadline);
        }
        self.instruction_count.store(0, Ordering::Relaxed);
        self.max_instructions.store(max_ops, Ordering::Relaxed);
        self.interrupted.store(false, Ordering::SeqCst);
    }

    /// Clears the active deadline upon completion.
    pub fn finish(&self) {
        if let Ok(mut g) = self.deadline.lock() {
            *g = None;
        }
        self.max_instructions.store(0, Ordering::Relaxed);
    }

    /// Returns `true` if execution was interrupted due to timeout or instruction fuel exhaustion.
    #[must_use]
    pub fn was_interrupted(&self) -> bool {
        self.interrupted.load(Ordering::SeqCst)
    }

    /// Returns total instructions/ticks counted in the last invocation.
    #[must_use]
    pub fn instructions_executed(&self) -> u64 {
        self.instruction_count.load(Ordering::Relaxed)
    }
}

impl Default for ExecutionController {
    fn default() -> Self {
        Self::new()
    }
}

/// A sandboxed JavaScript runtime environment.
pub struct JsSandbox {
    _runtime: Runtime,
    context: Context,
    controller: ExecutionController,
    config: JsSandboxConfig,
}

impl JsSandbox {
    /// Creates a new `JsSandbox` with the specified configuration.
    pub fn new(config: JsSandboxConfig) -> ModResult<Self> {
        let runtime = Runtime::new().map_err(|e| ModError::JsExecution {
            mod_id: "init".into(),
            message: format!("Failed to create QuickJS runtime: {e}"),
        })?;

        // Enforce memory and stack limits
        runtime.set_memory_limit(config.memory_limit);
        runtime.set_max_stack_size(config.max_stack_size);

        let controller = ExecutionController::new();
        let deadline_arc = controller.deadline.clone();
        let counter_arc = controller.instruction_count.clone();
        let max_arc = controller.max_instructions.clone();
        let interrupted_arc = controller.interrupted.clone();

        // Install interrupt handler for instruction fuel and timeout protection
        runtime.set_interrupt_handler(Some(Box::new(move || {
            let count = counter_arc.fetch_add(1, Ordering::Relaxed);
            let max = max_arc.load(Ordering::Relaxed);
            if max > 0 && count >= max {
                interrupted_arc.store(true, Ordering::SeqCst);
                return true; // Interrupt interpreter immediately
            }
            if let Ok(g) = deadline_arc.lock()
                && let Some(deadline) = *g
                && Instant::now() >= deadline
            {
                interrupted_arc.store(true, Ordering::SeqCst);
                return true; // Interrupt interpreter immediately
            }
            false
        })));

        // Create standard ECMAScript context
        let context = Context::full(&runtime).map_err(|e| ModError::JsExecution {
            mod_id: "init".into(),
            message: format!("Failed to create QuickJS context: {e}"),
        })?;

        // Setup safe globals (console logging routed to tracing)
        context.with(|ctx| -> ModResult<()> {
            let globals = ctx.globals();
            let console = Object::new(ctx.clone()).map_err(|e| ModError::JsExecution {
                mod_id: "init".into(),
                message: format!("Failed to create console object: {e}"),
            })?;

            let log_fn = Function::new(ctx.clone(), |msg: String| {
                tracing::info!(target: "telos::mod::js", "{msg}");
            })
            .map_err(|e| ModError::JsExecution {
                mod_id: "init".into(),
                message: format!("Failed to bind console.log: {e}"),
            })?;

            let warn_fn = Function::new(ctx.clone(), |msg: String| {
                tracing::warn!(target: "telos::mod::js", "{msg}");
            })
            .map_err(|e| ModError::JsExecution {
                mod_id: "init".into(),
                message: format!("Failed to bind console.warn: {e}"),
            })?;

            let error_fn = Function::new(ctx, |msg: String| {
                tracing::error!(target: "telos::mod::js", "{msg}");
            })
            .map_err(|e| ModError::JsExecution {
                mod_id: "init".into(),
                message: format!("Failed to bind console.error: {e}"),
            })?;

            console
                .set("log", log_fn)
                .map_err(|e| ModError::JsExecution {
                    mod_id: "init".into(),
                    message: format!("Failed to set console.log: {e}"),
                })?;
            console
                .set("warn", warn_fn)
                .map_err(|e| ModError::JsExecution {
                    mod_id: "init".into(),
                    message: format!("Failed to set console.warn: {e}"),
                })?;
            console
                .set("error", error_fn)
                .map_err(|e| ModError::JsExecution {
                    mod_id: "init".into(),
                    message: format!("Failed to set console.error: {e}"),
                })?;

            globals
                .set("console", console)
                .map_err(|e| ModError::JsExecution {
                    mod_id: "init".into(),
                    message: format!("Failed to register global console: {e}"),
                })?;

            Ok(())
        })?;

        Ok(Self {
            _runtime: runtime,
            context,
            controller,
            config,
        })
    }

    /// Evaluates a JavaScript snippet within the sandbox enforcing execution budgets.
    pub fn eval<T: for<'js> rquickjs::FromJs<'js>>(&self, code: &str) -> ModResult<T> {
        self.controller
            .begin(self.config.timeout_duration, self.config.max_instructions);

        let result = self
            .context
            .with(|ctx| -> Result<T, rquickjs::Error> { ctx.eval(code) });

        let interrupted = self.controller.was_interrupted();
        self.controller.finish();

        if interrupted {
            return Err(ModError::JsTimeout {
                mod_id: "eval".into(),
                message: format!(
                    "Execution exceeded budget (limit: {} instructions, {:?} timeout)",
                    self.config.max_instructions, self.config.timeout_duration
                ),
            });
        }

        result.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("out of memory") || err_str.contains("allocation") {
                ModError::JsMemoryLimitExceeded {
                    mod_id: "eval".into(),
                }
            } else {
                ModError::JsExecution {
                    mod_id: "eval".into(),
                    message: err_str,
                }
            }
        })
    }

    /// Loads and evaluates a named script file/content into the global environment.
    pub fn load_script(&self, name: &str, code: &str) -> ModResult<()> {
        self.controller
            .begin(self.config.timeout_duration, self.config.max_instructions);

        let result = self.context.with(|ctx| -> Result<(), rquickjs::Error> {
            let _: rquickjs::Value = ctx.eval(code)?;
            Ok(())
        });

        let interrupted = self.controller.was_interrupted();
        self.controller.finish();

        if interrupted {
            return Err(ModError::JsTimeout {
                mod_id: name.to_string(),
                message: format!("Script '{name}' exceeded execution budget during initial load"),
            });
        }

        result.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("out of memory") || err_str.contains("allocation") {
                ModError::JsMemoryLimitExceeded {
                    mod_id: name.to_string(),
                }
            } else {
                ModError::JsExecution {
                    mod_id: name.to_string(),
                    message: err_str,
                }
            }
        })
    }

    /// Returns a reference to the sandbox configuration.
    #[must_use]
    pub fn config(&self) -> &JsSandboxConfig {
        &self.config
    }

    /// Returns a reference to the underlying execution controller.
    #[must_use]
    pub fn controller(&self) -> &ExecutionController {
        &self.controller
    }

    /// Direct access to execute a closure within the underlying `QuickJS` context.
    pub fn with_context<F, R>(&self, f: F) -> ModResult<R>
    where
        F: FnOnce(rquickjs::Ctx<'_>) -> Result<R, rquickjs::Error>,
    {
        self.controller
            .begin(self.config.timeout_duration, self.config.max_instructions);

        let result = self.context.with(f);

        let interrupted = self.controller.was_interrupted();
        self.controller.finish();

        if interrupted {
            return Err(ModError::JsTimeout {
                mod_id: "context".into(),
                message: "Execution exceeded runtime budget".into(),
            });
        }

        result.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("out of memory") || err_str.contains("allocation") {
                ModError::JsMemoryLimitExceeded {
                    mod_id: "context".into(),
                }
            } else {
                ModError::JsExecution {
                    mod_id: "context".into(),
                    message: err_str,
                }
            }
        })
    }
}
