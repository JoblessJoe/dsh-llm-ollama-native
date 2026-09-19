# dsh-llm-ollama-native

An LLM adapter for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (dsh) that talks to Ollama's native `/api/chat` endpoint instead of its OpenAI-compatible one.

## Why this exists

If you run dsh against a local Ollama model and set `reasoningEffort` to `low`, `medium`, or `off`, you would expect the model to actually use less (or no) chain-of-thought. Through dsh's default Ollama route, it does not.

dsh's built-in Ollama support goes through `llm-pi-ai`, which talks to Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`). That endpoint silently ignores every reasoning-control field: `chat_template_kwargs`, a top-level `think`, a top-level `reasoning_effort`, streaming or not. The request always reaches the model with no signal, so the model falls back to its own default, which for most thinking models is the most verbose setting available.

This is not a dsh misconfiguration. It is an open, unresolved Ollama bug: [ollama/ollama#16240](https://github.com/ollama/ollama/issues/16240) tracks the same `chat_template_kwargs` being dropped by the OpenAI-compatible endpoint. As of writing there is no fix or workaround from upstream through that endpoint.

Ollama's *native* `/api/chat` endpoint does not have this problem. Its `think` field (`false`, `true`, or `"low"` / `"medium"` / `"high"` for models whose template supports it) reliably reaches the model. This package is a small adapter that uses that endpoint instead, so `reasoningEffort` in your dsh config does what it says.

If you are not on Ollama, or you already get correct behavior from your provider, you do not need this.

## Requirements

- dsh installed and configured to run against a local Ollama instance
- Ollama, with a model that supports `think` (check with `ollama show <model>`, capability `thinking`)
- Node 22 or newer

## Install

From the profile directory you want this in (for example `~/.dsh/profiles/<your-profile>`):

```sh
pnpm add dsh-llm-ollama-native@github:joblessjoe/dsh-llm-ollama-native
```

Add it to that profile's `package.json`, under `dsh.profile.bundles`:

```json
{
  "dsh": {
    "profile": {
      "bundles": [
        "...",
        "dsh-llm-ollama-native"
      ]
    }
  }
}
```

Then configure it in that profile's `cordis.patch.yml`:

```yaml
- id: llm-ollama-native
  config:
    provider: ollama-native
    baseURL: http://127.0.0.1:11434
    models:
      - id: your-model-tag
        contextWindow: 32768
        defaultReasoningEffort: low
```

Point your default model at the new route in `~/.dsh/settings.yaml`:

```yaml
agent-default-model:
  provider: ollama-native
  model: your-model-tag
  reasoningEffort: low
```

Restart dsh to pick up the new bundle. `settings.yaml` itself hot-reloads, but adding a plugin to `dsh.profile.bundles` needs a restart.

Your existing `ollama` route through `llm-pi-ai` is untouched by any of this — switch `provider` back to `ollama` at any time to revert.

## Verify

`test.js` drives the adapter directly against a running Ollama server (no dsh profile needs to be running, though `@deepseek-ai/dsh-llm` — the type/base-class package the adapter extends — does need to be installed):

```sh
npm install
node test.js
```

It checks that `reasoningEffort: "off"` produces no reasoning output, that `"low"` produces some, and that a tool call round-trips correctly. Defaults to `qwen3.8:27b`; point it at a different model with `OLLAMA_TEST_MODEL=your-model:tag node test.js`.

## Configuration reference

| Field | Required | Description |
|---|---|---|
| `provider` | yes | Route name other dsh config refers to (e.g. `agent-default-model.provider`) |
| `baseURL` | no | Ollama server address, default `http://localhost:11434` |
| `displayName` | no | Shown in model pickers, default `Ollama (native)` |
| `models` | yes | Array of model entries |
| `models[].id` | yes | Ollama model tag |
| `models[].contextWindow` | no | Context window size, for display and request sizing |
| `models[].defaultReasoningEffort` | no | `off`, `low`, `medium`, or `high`; used when a request specifies none, default `low` |

## Limitations

This is a minimal adapter, not a full port of `llm-pi-ai`:

- No image or file content blocks; text and reasoning blocks only
- No replay of prior assistant reasoning into follow-up requests
- No retry policy or provider-side image pricing
- Graduated reasoning levels (`low` / `medium` / `high`) depend on your model's own chat template supporting them; `off` is the one guaranteed to work everywhere, since it is a hard switch in Ollama itself

## License

MIT
