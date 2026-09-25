// Runnable check: drives OllamaNativeAdapter.stream() directly against a
// live Ollama server (no DSH runtime needed) and asserts the one thing this
// package exists for — reasoningEffort actually reaches the model.
//
//   node test.js
//   OLLAMA_TEST_MODEL=your-model:tag node test.js
//
// Requires Ollama running locally with a "thinking"-capable model pulled
// (check with `ollama show <model>`). Defaults to qwen3.8:27b.

import assert from 'node:assert/strict'
import { LlmAdapter } from '@deepseek-ai/dsh-llm'
import { OllamaNativeAdapter } from './adapter.js'

const MODEL = process.env.OLLAMA_TEST_MODEL ?? 'qwen3.8:27b'

const adapter = new OllamaNativeAdapter({
  provider: 'ollama-native',
  models: [{ id: MODEL, contextWindow: 77824, defaultReasoningEffort: 'low' }],
})

/** @param {string} text */
function userMessage(text) {
  return { id: 'm1', role: 'user', content: [{ type: 'text', text }], source: { kind: 'user' } }
}

/** Drain a stream(), returning the accumulated reasoning text, visible text, and any tool calls. */
async function run(options, adapterOverride = adapter) {
  let reasoning = ''
  let text = ''
  const toolCalls = []
  let finish
  for await (const chunk of adapterOverride.stream(options)) {
    if (chunk.type === 'reasoning-delta') reasoning += chunk.text
    if (chunk.type === 'text-delta') text += chunk.text
    if (chunk.type === 'block-end' && chunk.block.type === 'tool-call') toolCalls.push(chunk.block)
    if (chunk.type === 'finish') finish = chunk.reason
  }
  return { reasoning, text, toolCalls, finish }
}

async function main() {
  console.log('0) extends the real LlmAdapter (not a hand-maintained duck-type copy)...')
  // A previous version hand-implemented this interface instead of extending
  // the real class, and silently missed one method (imageRequestPricing) —
  // dsh's token-meter calls it unconditionally, not optionally-chained, so
  // the gap crashed compaction on every attempt. Extending the real class
  // means a future default method dsh adds is inherited automatically
  // instead of needing to be remembered and kept in sync here by hand.
  assert.ok(adapter instanceof LlmAdapter, 'OllamaNativeAdapter must extend the real LlmAdapter')
  console.log('   OK')

  console.log('1) reasoningEffort "off" must fully suppress thinking...')
  const off = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'off',
    messages: [userMessage('What is 2+2? Answer in one word.')],
  })
  assert.equal(off.reasoning, '', `expected no reasoning with effort "off", got: ${off.reasoning}`)
  console.log('   OK — reasoning:', JSON.stringify(off.reasoning), 'text:', JSON.stringify(off.text))

  console.log('2) reasoningEffort "low" must produce some reasoning (proves the field reaches the model)...')
  const low = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'low',
    messages: [userMessage('What is 2+2? Answer in one word.')],
  })
  assert.ok(low.reasoning.length > 0, 'expected non-empty reasoning with effort "low"')
  console.log('   OK — reasoning length:', low.reasoning.length, 'text:', JSON.stringify(low.text))

  console.log('3) maxTokens must actually cap output (was silently dropped before)...')
  const capped = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'off',
    maxTokens: 5,
    messages: [userMessage('Write a 300 word essay about the ocean.')],
  })
  assert.equal(capped.finish.kind, 'max-tokens', `expected finish reason "max-tokens", got "${capped.finish?.kind}" — maxTokens isn't reaching Ollama`)
  console.log('   OK — finish reason:', capped.finish.kind, 'text len:', capped.text.length)

  console.log('4) tool call round-trip...')
  const toolCall = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'low',
    messages: [userMessage('What is the weather in Berlin? Use the tool.')],
    tools: [{
      name: 'get_weather',
      description: 'Get weather for a city',
      parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    }],
  })
  assert.equal(toolCall.toolCalls.length, 1, 'expected exactly one tool call')
  assert.equal(toolCall.toolCalls[0].name, 'get_weather')
  const args = JSON.parse(toolCall.toolCalls[0].arguments)
  assert.equal(args.city, 'Berlin')
  assert.equal(toolCall.finish.kind, 'tool-calls')
  console.log('   OK — tool call:', toolCall.toolCalls[0].name, args)

  console.log('5) thinkingBudgetTokens must cut off a runaway reasoning attempt and still return a real answer...')
  const budgetedAdapter = new OllamaNativeAdapter({
    provider: 'ollama-native',
    // A tiny budget (~40 tokens) makes this test fast and forces the cutover
    // on almost any prompt, instead of needing a prompt that reliably
    // reasons past a realistic production budget (~1.5-2k).
    models: [{ id: MODEL, contextWindow: 77824, defaultReasoningEffort: 'high', thinkingBudgetTokens: 40 }],
  })
  const budgeted = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'high',
    messages: [userMessage(
      'Compare two ways to cache API responses in a Node service — an in-memory LRU vs a Redis-backed '
      + 'cache — and recommend one, with reasons.',
    )],
  }, budgetedAdapter)
  assert.ok(budgeted.text.length > 0, `expected a real final answer after the budget cutover, got empty text (finish: ${JSON.stringify(budgeted.finish)})`)
  assert.ok(
    budgeted.reasoning.includes('[thinking truncated — budget reached]'),
    'expected the truncation marker in the accumulated reasoning — budget cutover did not trigger',
  )
  assert.notEqual(budgeted.finish?.kind, 'error', `expected a clean finish after retry, got: ${JSON.stringify(budgeted.finish)}`)
  console.log('   OK — reasoning length:', budgeted.reasoning.length, 'text length:', budgeted.text.length, 'finish:', budgeted.finish.kind)

  console.log('6) thinkingBudgetTokens unset must leave existing behavior untouched (no cutover, no marker)...')
  const unbudgeted = await run({
    provider: 'ollama-native',
    model: MODEL,
    reasoningEffort: 'low',
    messages: [userMessage('What is 2+2? Answer in one word.')],
  })
  assert.ok(
    !unbudgeted.reasoning.includes('[thinking truncated'),
    'no budget configured — must never see a truncation marker',
  )
  console.log('   OK — reasoning:', JSON.stringify(unbudgeted.reasoning))

  console.log('\nAll checks passed.')
}

main().catch(error => {
  console.error('FAILED:', error)
  process.exitCode = 1
})
