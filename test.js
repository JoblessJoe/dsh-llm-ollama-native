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
async function run(options) {
  let reasoning = ''
  let text = ''
  const toolCalls = []
  let finish
  for await (const chunk of adapter.stream(options)) {
    if (chunk.type === 'reasoning-delta') reasoning += chunk.text
    if (chunk.type === 'text-delta') text += chunk.text
    if (chunk.type === 'block-end' && chunk.block.type === 'tool-call') toolCalls.push(chunk.block)
    if (chunk.type === 'finish') finish = chunk.reason
  }
  return { reasoning, text, toolCalls, finish }
}

async function main() {
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

  console.log('3) tool call round-trip...')
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

  console.log('\nAll checks passed.')
}

main().catch(error => {
  console.error('FAILED:', error)
  process.exitCode = 1
})
