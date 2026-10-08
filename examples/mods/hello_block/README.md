# Hello Block (`hello_block`)

Telos Engine Mod.

## Development & Testing
- Validate manifest and schemas:
  ```bash
  cargo xtask mod validate examples/mods/hello_block
  ```
- Build distribution package (`.vxmod` and `server.wasm`):
  ```bash
  cargo xtask mod build examples/mods/hello_block
  ```
