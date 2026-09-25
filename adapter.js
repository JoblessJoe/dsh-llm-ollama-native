/**
 * LlmAdapter that speaks Ollama's NATIVE /api/chat instead of the generic
 * OpenAI-compatible /v1/chat/completions endpoint.
 *
 * Root cause this exists for: Ollama's OpenAI-compat endpoint silently drops
 * every reasoning-control field (`chat_template_kwargs`, top-level `think`,
 * top-level `reasoning_effort`, nested `options.think`, streaming or not —
 * all verified dropped by direct curl against Ollama 0.34.2). Only the native
 * endpoint's `think` field (bool, or "low"/"medium"/"high" for a template
 * that supports it) actually reaches the model. This adapter exists purely
 * to reach that field; everything else about it is deliberately minimal.
 *
 * Known limitations (MVP scope, not a full port of llm-pi-ai):
 * - No image/file content blocks — text and reasoning blocks only.
 * - No replay of prior assistant reasoning into follow-up requests (each
 *   request's history sends only visible text + tool calls/results).
 * - No retry policy, no attribution headers.
 * - A model's `think` level (Rule 9 in ~/.dsh/AGENTS.md) is a system-prompt
 *   INSTRUCTION, not an enforced token budget — Ollama has no server-side
 *   ceiling on reasoning length (confirmed: ollama/ollama#17561). A model can
 *   still generate an arbitrarily long reasoning block regardless of effort.
 *   `thinkingBudgetTokens` below is this adapter's own client-side backstop
 *   for that gap.
 */

import { LlmAdapter } from '@deepseek-ai/dsh-llm'

/** @typedef {import('@deepseek-ai/dsh-llm').GenerateOptions} GenerateOptions */
/** @typedef {import('@deepseek-ai/dsh-llm').StreamChunk} StreamChunk */
/** @typedef {import('@deepseek-ai/dsh-llm').Message} Message */

/** DSH reasoning-effort id -> Ollama native `think` value. */
const THINK_BY_EFFORT = {
  off: false,
  low: 'low',
  medium: 'medium',
  high: 'high',
}

/**
 * One configured model route.
 * @typedef {object} OllamaNativeModel
 * @property {string} id - Ollama model tag, e.g. "qwen3.8:27b".
 * @property {string} [name] - display name; defaults to `id`.
 * @property {number} [contextWindow]
 * @property {keyof THINK_BY_EFFORT} [defaultReasoningEffort] - sent when a request omits one; defaults to "low" (the whole point of this adapter).
 * @property {boolean} [supportsThinking] - set false for a model whose chat template has no thinking support (verify via its GGUF-embedded chat_template: no `<think>`/`enable_thinking`). Ollama hard-errors ("%q does not support thinking") on ANY explicit `think` field for such a model, even `false` — omit the field entirely instead. Defaults to true.
 * @property {number} [thinkingBudgetTokens] - client-side backstop on reasoning length (rough chars/4 estimate, not real tokenization). Ollama's `think` level does not cap length on its own (ollama/ollama#17561) — a model can still run away. Unset = no backstop, current behavior unchanged. When reasoning crosses this estimate mid-think, this adapter aborts that attempt and re-issues once with `think:false` plus a short note of the truncated reasoning, so the caller still gets a real answer instead of an empty or runaway response. See Qwen's own documented two-call budget pattern (github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md) — this is a robustness-first adaptation of it built only on request shapes already verified working through this adapter (message roles, `think:false`), not on unverified assistant-turn continuation/prefill behavior.
 */

/**
 * Join a message's content blocks into the plain text Ollama's chat API
 * expects, dropping non-text blocks this MVP does not project.
 * @param {Message} message
 * @returns {string}
 */
function textOf(message) {
  return message.content
    .filter(block => block.type === 'text')
    .map(block => block.text)
    .join('')
}

/**
 * Project one DSH history message to Ollama's native `{role, content, ...}`
 * shape. A tool-result message (DSH: user-role, one ToolResultBlock) becomes
 * an Ollama `role: "tool"` message, matching this model's chat template.
 * @param {Message} message
 * @returns {Record<string, unknown>}
 */
function toOllamaMessage(message) {
  const toolResult = message.content.find(block => block.type === 'tool-result')
  if (toolResult !== undefined) {
    const text = toolResult.content
      .filter(block => block.type === 'text')
      .map(block => block.text)
      .join('')
    return { role: 'tool', content: text }
  }
  const toolCalls = message.content.filter(block => block.type === 'tool-call')
  return {
    role: message.role,
    content: textOf(message),
    ...toolCalls.length === 0 ? {} : {
      tool_calls: toolCalls.map(call => ({
        function: { name: call.name, arguments: JSON.parse(call.arguments) },
      })),
    },
  }
}

/** @param {import('@deepseek-ai/dsh-llm').ToolSchema[] | undefined} tools */
function toOllamaTools(tools) {
  if (tools === undefined || tools.length === 0) return undefined
  return tools.map(tool => ({
    type: 'function',
    function: { name: tool.name, description: tool.description, parameters: tool.parameters },
  }))
}

/**
 * Collapse consecutive same-role messages into one. DSH hands back the
 * human's message plus separately-tracked system-reminder/context/skill
 * blocks as several back-to-back `role: "user"` entries — harmless for a
 * lenient chat template (Qwen), but a strict one (Mistral/Devstral) raises
 * "conversation roles must alternate user and assistant... except for tool
 * calls and results" on anything but genuine alternation. Tool messages are
 * left alone: consecutive tool results are a normal, exempt shape.
 * @param {Record<string, unknown>[]} messages
 * @returns {Record<string, unknown>[]}
 */
function mergeConsecutiveSameRole(messages) {
  const merged = []
  for (const message of messages) {
    const prev = merged[merged.length - 1]
    if (prev !== undefined && prev.role === message.role && message.role !== 'tool') {
      prev.content = [prev.content, message.content].filter(text => text.length > 0).join('\n\n')
      if (Array.isArray(message.tool_calls)) {
        prev.tool_calls = [...Array.isArray(prev.tool_calls) ? prev.tool_calls : [], ...message.tool_calls]
      }
      continue
    }
    merged.push({ ...message })
  }
  return merged
}

/** @param {string} done_reason */
function finishReasonOf(done_reason, hasToolCalls) {
  if (hasToolCalls) return { kind: 'tool-calls' }
  if (done_reason === 'length') return { kind: 'max-tokens' }
  return { kind: 'stop' }
}

/**
 * Cheap, honest token estimate — chars/4, not real tokenization. This only
 * guards a client-side safety backstop ("stop before 20k, not before
 * exactly N"), so being off by 20-30% doesn't matter; a real tokenizer
 * would be a heavier dependency for no practical gain here.
 * @param {string} text
 * @returns {number}
 */
function estimateTokens(text) {
  return Math.ceil(text.length / 4)
}

/** How much of the truncated reasoning to carry into the retry's note. */
const RETRY_NOTE_REASONING_CHARS = 800

export class OllamaNativeAdapter extends LlmAdapter {
  /**
   * @param {object} options
   * @param {string} options.provider - route key this instance owns (registered by the caller).
   * @param {string} [options.baseURL] - Ollama server base, default `http://localhost:11434`.
   * @param {string} [options.displayName] - shown in model pickers; default `Ollama (native, <provider>)`,
   *   which stays unique across multiple routes on this adapter — only override if you want that.
   * @param {OllamaNativeModel[]} options.models
   */
  constructor(options) {
    super()
    this.provider = options.provider
    this.baseURL = options.baseURL ?? 'http://localhost:11434'
    this.displayName = options.displayName ?? `Ollama (native, ${options.provider})`
    /** @type {Map<string, OllamaNativeModel>} */
    this.models = new Map(options.models.map(model => [model.id, model]))
  }

  // extends LlmAdapter for providerRetryPolicy/imageRequestPricing/prepareCall's
  // working defaults — a hand-rolled duck-typed copy of this class silently
  // dropped one of them (imageRequestPricing) and crashed compaction on every
  // attempt; extending the real class means a future default method dsh adds
  // is inherited automatically instead of needing to be remembered here.

  providerInfo() {
    return { id: this.provider, name: this.displayName }
  }

  async listModels() {
    return [...this.models.values()].map(model => ({
      provider: this.provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: ['text'],
    }))
  }

  /** @param {string} _provider @param {string} model */
  async resolveModel(_provider, model) {
    const entry = this.models.get(model)
    if (entry === undefined) {
      throw new Error(`dsh-llm-ollama-native: no configured model "${model}" on provider "${this.provider}"`)
    }
    const defaultEffort = entry.defaultReasoningEffort ?? 'low'
    return {
      provider: this.provider,
      id: entry.id,
      name: entry.name ?? entry.id,
      inputModalities: ['text'],
      ...entry.contextWindow === undefined ? {} : { context: { contextWindow: entry.contextWindow } },
      reasoning: {
        efforts: Object.keys(THINK_BY_EFFORT).map(id => ({ id, name: id[0].toUpperCase() + id.slice(1) })),
        defaultEffort,
      },
    }
  }

  // prepareCall: inherited from LlmAdapter — its default (resolveModel + a
  // stream() closure) is exactly what this adapter needs, no override.

  /**
   * Run one /api/chat attempt, translating Ollama's native events to
   * StreamChunks at the given block-index offset (lets a retry attempt open
   * fresh blocks without colliding with indices the first attempt already
   * used — see `stream()`).
   *
   * Returns (as the generator's final value, not a yielded chunk):
   * - `{ hitBudget: true, reasoningSoFar }` if reasoning crossed
   *   `budgetTokens` before the model finished thinking. The attempt is
   *   aborted at that point and a real `finish` is NOT yielded — the caller
   *   is expected to retry.
   * - `{ hitBudget: false }` on a normal finish, a real external abort, or a
   *   dropped connection — in every one of those cases this method already
   *   yielded the terminal `finish` chunk itself, same as before this
   *   backstop existed.
   *
   * @param {Record<string, unknown>} body
   * @param {number | undefined} budgetTokens
   * @param {AbortSignal | undefined} externalSignal
   * @param {number} indexOffset
   */
  async * #streamAttempt(body, budgetTokens, externalSignal, indexOffset) {
    const controller = new AbortController()
    let budgetAbort = false
    const forwardAbort = () => controller.abort()
    if (externalSignal !== undefined) {
      if (externalSignal.aborted) controller.abort()
      else externalSignal.addEventListener('abort', forwardAbort, { once: true })
    }

    const reasoningIndex = indexOffset
    const textIndex = indexOffset + 1
    const toolCallIndexBase = indexOffset + 2

    let reasoningOpen = false
    let reasoningText = ''
    let textOpen = false
    let textAccum = ''

    try {
      const response = await fetch(`${this.baseURL}/api/chat`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
        signal: controller.signal,
      })
      if (!response.ok || response.body === null) {
        const text = await response.text().catch(() => '')
        throw new Error(`dsh-llm-ollama-native: ${response.status} ${response.statusText} ${text}`)
      }

      let buffer = ''
      let usageSent = false
      const decoder = new TextDecoder('utf-8')

      for await (const chunk of response.body) {
        buffer += decoder.decode(chunk, { stream: true })
        let newline
        while ((newline = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, newline)
          buffer = buffer.slice(newline + 1)
          if (line.trim().length === 0) continue
          const event = JSON.parse(line)
          const message = event.message ?? {}

          if (typeof message.thinking === 'string' && message.thinking.length > 0) {
            if (!reasoningOpen) {
              reasoningOpen = true
              yield { type: 'block-start', index: reasoningIndex, blockType: 'reasoning' }
            }
            reasoningText += message.thinking
            yield { type: 'reasoning-delta', index: reasoningIndex, text: message.thinking }

            if (budgetTokens !== undefined && estimateTokens(reasoningText) >= budgetTokens) {
              const note = '\n\n[thinking truncated — budget reached]'
              reasoningText += note
              yield { type: 'reasoning-delta', index: reasoningIndex, text: note }
              yield { type: 'block-end', index: reasoningIndex, block: { type: 'reasoning', text: reasoningText } }
              budgetAbort = true
              controller.abort()
              return { hitBudget: true, reasoningSoFar: reasoningText }
            }
          }
          if (typeof message.content === 'string' && message.content.length > 0) {
            if (!textOpen) {
              textOpen = true
              yield { type: 'block-start', index: textIndex, blockType: 'text' }
            }
            textAccum += message.content
            yield { type: 'text-delta', index: textIndex, text: message.content }
          }

          if (event.done === true) {
            if (reasoningOpen) {
              yield { type: 'block-end', index: reasoningIndex, block: { type: 'reasoning', text: reasoningText } }
            }
            if (textOpen) {
              yield { type: 'block-end', index: textIndex, block: { type: 'text', text: textAccum } }
            }
            const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : []
            for (const [position, call] of toolCalls.entries()) {
              const index = toolCallIndexBase + position
              const id = call.id ?? `${Date.now()}-${position}`
              const argumentsJson = JSON.stringify(call.function?.arguments ?? {})
              yield { type: 'block-start', index, blockType: 'tool-call' }
              yield {
                type: 'tool-call-delta',
                index,
                id,
                name: call.function?.name,
                argumentsDelta: argumentsJson,
              }
              yield {
                type: 'block-end',
                index,
                block: { type: 'tool-call', id, name: call.function?.name ?? '', arguments: argumentsJson },
              }
            }
            if (!usageSent && (event.prompt_eval_count !== undefined || event.eval_count !== undefined)) {
              usageSent = true
              yield {
                type: 'usage',
                usage: {
                  inputTokens: event.prompt_eval_count ?? 0,
                  outputTokens: event.eval_count ?? 0,
                },
              }
            }
            yield { type: 'finish', reason: finishReasonOf(event.done_reason, toolCalls.length > 0) }
            return { hitBudget: false }
          }
        }
      }
    } catch (error) {
      if (budgetAbort) {
        // We aborted this ourselves to cut over to the retry — not a real
        // failure, swallow it. The caller yields no finish for this attempt.
        return { hitBudget: true, reasoningSoFar: reasoningText }
      }
      if (!(externalSignal?.aborted === true)) throw error
      // else: a genuine external cancellation raced the fetch/stream — fall
      // through to the same terminal-chunk logic as a body that closed
      // early, same as pre-existing behavior.
    } finally {
      if (externalSignal !== undefined) externalSignal.removeEventListener('abort', forwardAbort)
    }

    // The response body closed without ever sending a `done:true` line, and
    // this was not our own budget-triggered abort — Ollama crashed, the
    // connection dropped, the process was killed, or the caller's own
    // signal aborted mid-generation. Every stream must end in a `finish`
    // chunk (dsh's own invariant checker rejects one that doesn't with an
    // opaque internal error instead of a clean, callable failure) — close
    // whatever blocks were open and report it as the real, specific
    // failure it is.
    if (reasoningOpen) yield { type: 'block-end', index: reasoningIndex, block: { type: 'reasoning', text: reasoningText } }
    if (textOpen) yield { type: 'block-end', index: textIndex, block: { type: 'text', text: textAccum } }
    const failure = { message: 'Ollama closed the connection before sending a final response', code: 'STREAM_ENDED_EARLY' }
    yield {
      type: 'finish',
      reason: externalSignal?.aborted === true ? { kind: 'aborted', failure } : { kind: 'error', failure },
    }
    return { hitBudget: false }
  }

  /**
   * @param {GenerateOptions} options
   * @returns {AsyncIterable<StreamChunk>}
   */
  async * stream(options) {
    const entry = this.models.get(options.model)
    const effort = options.reasoningEffort ?? entry?.defaultReasoningEffort ?? 'low'
    const think = THINK_BY_EFFORT[effort] ?? 'low'
    const budgetTokens = entry?.thinkingBudgetTokens

    const messages = mergeConsecutiveSameRole([
      ...options.system === undefined || options.system.length === 0
        ? []
        : [{ role: 'system', content: options.system }],
      ...options.messages.map(toOllamaMessage),
    ])

    const sampling = {
      ...options.temperature === undefined ? {} : { temperature: options.temperature },
      // Ollama's native name for the output-token cap ("maxTokens" in dsh's
      // provider-neutral vocabulary). Missing this meant every call generated
      // uncapped — harmless for ordinary turns, but silently starved
      // compaction's summarization call of the room it needed within a
      // near-full context, surfacing as "summarization truncated at the
      // token cap" with no indication the cap was never actually sent.
      ...options.maxTokens === undefined ? {} : { num_predict: options.maxTokens },
    }

    const baseBody = {
      model: options.model,
      messages,
      stream: true,
      ...Object.keys(sampling).length === 0 ? {} : { options: sampling },
      ...toOllamaTools(options.tools) === undefined ? {} : { tools: toOllamaTools(options.tools) },
    }

    const firstBody = { ...baseBody, ...entry?.supportsThinking === false ? {} : { think } }

    const attempt1 = this.#streamAttempt(firstBody, budgetTokens, options.signal, 0)
    const it1 = attempt1[Symbol.asyncIterator]()
    let result1
    while (true) {
      const next = await it1.next()
      if (next.done) { result1 = next.value; break }
      yield next.value
    }
    if (!result1.hitBudget) return

    // Budget retry: one extra shot, reasoning forced off, with a short note
    // of what the truncated reasoning had gotten to so the retry isn't
    // starting from nothing. Deliberately NOT attempting to make the model
    // "continue" the cut-off <think> block via an assistant-role prefill —
    // that depends on this exact GGUF's chat-template Jinja treating a
    // trailing assistant message as continuation rather than a new turn,
    // which is not verified for this model. A plain system-role note plus
    // `think:false` only relies on request shapes this adapter already
    // proves work correctly.
    const retryNote = {
      role: 'system',
      content: `Your reasoning on this turn ran long and was cut off at the configured thinking budget. `
        + `Partial reasoning notes (may be incomplete): ${result1.reasoningSoFar.slice(0, RETRY_NOTE_REASONING_CHARS)}\n\n`
        + 'Answer directly now — extended step-by-step reasoning is disabled for this retry.',
    }
    const retryBody = {
      ...baseBody,
      messages: [...messages, retryNote],
      ...entry?.supportsThinking === false ? {} : { think: false },
    }

    const attempt2 = this.#streamAttempt(retryBody, undefined, options.signal, 2)
    const it2 = attempt2[Symbol.asyncIterator]()
    while (true) {
      const next = await it2.next()
      if (next.done) return
      yield next.value
    }
  }
}
