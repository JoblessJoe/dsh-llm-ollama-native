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
 * - No retry policy, no attribution headers, no image pricing.
 */

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

/** @param {string} done_reason */
function finishReasonOf(done_reason, hasToolCalls) {
  if (hasToolCalls) return { kind: 'tool-calls' }
  if (done_reason === 'length') return { kind: 'max-tokens' }
  return { kind: 'stop' }
}

export class OllamaNativeAdapter {
  /**
   * @param {object} options
   * @param {string} options.provider - route key this instance owns (registered by the caller).
   * @param {string} [options.baseURL] - Ollama server base, default `http://localhost:11434`.
   * @param {string} [options.displayName] - shown in model pickers; default `Ollama (native)`.
   * @param {OllamaNativeModel[]} options.models
   */
  constructor(options) {
    this.provider = options.provider
    this.baseURL = options.baseURL ?? 'http://localhost:11434'
    this.displayName = options.displayName ?? 'Ollama (native)'
    /** @type {Map<string, OllamaNativeModel>} */
    this.models = new Map(options.models.map(model => [model.id, model]))
  }

  providerInfo() {
    return { id: this.provider, name: this.displayName }
  }

  providerRetryPolicy() {
    return undefined
  }

  /** No image support (see README limitations) — no per-image pricing to report. */
  imageRequestPricing() {
    return undefined
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

  async prepareCall(provider, model, signal) {
    return { model: await this.resolveModel(provider, model, signal), stream: options => this.stream(options) }
  }

  /**
   * @param {GenerateOptions} options
   * @returns {AsyncIterable<StreamChunk>}
   */
  async * stream(options) {
    const entry = this.models.get(options.model)
    const effort = options.reasoningEffort ?? entry?.defaultReasoningEffort ?? 'low'
    const think = THINK_BY_EFFORT[effort] ?? 'low'

    const messages = [
      ...options.system === undefined || options.system.length === 0
        ? []
        : [{ role: 'system', content: options.system }],
      ...options.messages.map(toOllamaMessage),
    ]

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

    const body = {
      model: options.model,
      messages,
      think,
      stream: true,
      ...Object.keys(sampling).length === 0 ? {} : { options: sampling },
      ...toOllamaTools(options.tools) === undefined ? {} : { tools: toOllamaTools(options.tools) },
    }

    const response = await fetch(`${this.baseURL}/api/chat`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: options.signal,
    })
    if (!response.ok || response.body === null) {
      const text = await response.text().catch(() => '')
      throw new Error(`dsh-llm-ollama-native: ${response.status} ${response.statusText} ${text}`)
    }

    // Index 0 = reasoning block, 1 = text block, 2.. = tool-call blocks —
    // fixed slots, opened lazily on first content, matching StreamChunk's
    // "block-start once, then deltas, then block-end" contract.
    let reasoningOpen = false
    let reasoningText = ''
    let textOpen = false
    let textAccum = ''
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
            yield { type: 'block-start', index: 0, blockType: 'reasoning' }
          }
          reasoningText += message.thinking
          yield { type: 'reasoning-delta', index: 0, text: message.thinking }
        }
        if (typeof message.content === 'string' && message.content.length > 0) {
          if (!textOpen) {
            textOpen = true
            yield { type: 'block-start', index: 1, blockType: 'text' }
          }
          textAccum += message.content
          yield { type: 'text-delta', index: 1, text: message.content }
        }

        if (event.done === true) {
          if (reasoningOpen) {
            yield { type: 'block-end', index: 0, block: { type: 'reasoning', text: reasoningText } }
          }
          if (textOpen) {
            yield { type: 'block-end', index: 1, block: { type: 'text', text: textAccum } }
          }
          const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : []
          for (const [position, call] of toolCalls.entries()) {
            const index = 2 + position
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
          return
        }
      }
    }
  }
}
