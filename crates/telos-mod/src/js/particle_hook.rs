//! Client-side dynamic particle and sound modification hooks powered by JavaScript.

use rquickjs::{Function, Object, Value};

use super::sandbox::{JsSandbox, JsSandboxConfig};
use crate::error::ModResult;

/// Particle simulation parameters passed to and returned from JavaScript hooks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JsParticleParams {
    /// World position (x, y, z).
    pub pos: [f32; 3],
    /// Linear velocity (vx, vy, vz).
    pub velocity: [f32; 3],
    /// Color tint in 0xAABBGGRR / 0xRRGGBBAA packed format.
    pub color_tint: u32,
    /// Visual quad scale factor.
    pub scale: f32,
    /// Lifespan duration in frames/ticks.
    pub lifetime: f32,
}

impl Default for JsParticleParams {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            velocity: [0.0; 3],
            color_tint: 0xFFFF_FFFF,
            scale: 1.0,
            lifetime: 20.0,
        }
    }
}

/// Dynamic particle hook evaluator.
pub struct JsParticleHook {
    sandbox: JsSandbox,
    has_spawn_hook: bool,
    has_sound_hook: bool,
}

impl JsParticleHook {
    /// Creates a new `JsParticleHook` from a script source string.
    pub fn new(script_source: &str) -> ModResult<Self> {
        let sandbox = JsSandbox::new(JsSandboxConfig::default())?;
        sandbox.load_script("particle_effects.js", script_source)?;

        let (has_spawn_hook, has_sound_hook) = sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let has_spawn = globals
                .get::<_, Value>("onParticleSpawn")
                .is_ok_and(|v| v.is_function());
            let has_sound = globals
                .get::<_, Value>("onSoundPlay")
                .is_ok_and(|v| v.is_function());
            Ok((has_spawn, has_sound))
        })?;

        Ok(Self {
            sandbox,
            has_spawn_hook,
            has_sound_hook,
        })
    }

    /// Evaluates `onParticleSpawn(info)` and returns modified particle parameters.
    #[must_use]
    pub fn on_particle_spawn(
        &self,
        particle_type: &str,
        mut params: JsParticleParams,
    ) -> JsParticleParams {
        if !self.has_spawn_hook {
            return params;
        }

        let res: Result<JsParticleParams, _> = self.sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let func: Function = globals.get("onParticleSpawn")?;

            let arg = Object::new(ctx)?;
            arg.set("type", particle_type)?;
            arg.set("x", f64::from(params.pos[0]))?;
            arg.set("y", f64::from(params.pos[1]))?;
            arg.set("z", f64::from(params.pos[2]))?;
            arg.set("vx", f64::from(params.velocity[0]))?;
            arg.set("vy", f64::from(params.velocity[1]))?;
            arg.set("vz", f64::from(params.velocity[2]))?;
            arg.set("color", params.color_tint)?;
            arg.set("scale", f64::from(params.scale))?;
            arg.set("lifetime", f64::from(params.lifetime))?;

            let ret: Value = func.call((arg,))?;
            if let Some(obj) = ret.as_object() {
                if let Ok(vx) = obj.get::<_, f64>("vx") {
                    params.velocity[0] = vx as f32;
                }
                if let Ok(vy) = obj.get::<_, f64>("vy") {
                    params.velocity[1] = vy as f32;
                }
                if let Ok(vz) = obj.get::<_, f64>("vz") {
                    params.velocity[2] = vz as f32;
                }
                if let Ok(color) = obj.get::<_, u32>("color") {
                    params.color_tint = color;
                }
                if let Ok(scale) = obj.get::<_, f64>("scale") {
                    params.scale = scale as f32;
                }
                if let Ok(lifetime) = obj.get::<_, f64>("lifetime") {
                    params.lifetime = lifetime as f32;
                }
            }

            Ok(params)
        });

        res.unwrap_or(params)
    }

    /// Evaluates `onSoundPlay(sound_id, volume, pitch)` returning modified (volume, pitch).
    #[must_use]
    pub fn on_sound_play(&self, sound_id: &str, volume: f32, pitch: f32) -> (f32, f32) {
        if !self.has_sound_hook {
            return (volume, pitch);
        }

        let res: Result<(f32, f32), _> = self.sandbox.with_context(|ctx| {
            let globals = ctx.globals();
            let func: Function = globals.get("onSoundPlay")?;

            let arg = Object::new(ctx)?;
            arg.set("id", sound_id)?;
            arg.set("volume", f64::from(volume))?;
            arg.set("pitch", f64::from(pitch))?;

            let ret: Value = func.call((arg,))?;
            let mut out_vol = volume;
            let mut out_pitch = pitch;

            if let Some(obj) = ret.as_object() {
                if let Ok(v) = obj.get::<_, f64>("volume") {
                    out_vol = (v as f32).clamp(0.0, 2.0);
                }
                if let Ok(p) = obj.get::<_, f64>("pitch") {
                    out_pitch = (p as f32).clamp(0.1, 2.0);
                }
            }

            Ok((out_vol, out_pitch))
        });

        res.unwrap_or((volume, pitch))
    }

    /// Returns `true` if this hook contains an `onParticleSpawn` callback.
    #[must_use]
    pub fn has_particle_hook(&self) -> bool {
        self.has_spawn_hook
    }

    /// Returns `true` if this hook contains an `onSoundPlay` callback.
    #[must_use]
    pub fn has_sound_hook(&self) -> bool {
        self.has_sound_hook
    }
}
