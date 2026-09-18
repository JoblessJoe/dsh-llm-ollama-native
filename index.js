/**
 * Cordis plugin wrapper: registers {@link OllamaNativeAdapter} on `ctx.llm`
 * for one provider route. Not needed to test the adapter itself (see
 * `test.js`, which drives `adapter.js` directly against a live Ollama
 * server) — only needed once this is wired into a dsh profile.
 *
 * This package declares itself as a dsh bundle (`dsh.bundle.patch` in
 * package.json), so once it is in a profile's `dsh.profile.bundles`, it is
 * already mounted under id `llm-ollama-native` with an empty model list.
 * A profile's own `cordis.patch.yml` then just patches that id's config —
 * see README.md for the full install steps.
 */

import { OllamaNativeAdapter } from './adapter.js'

export const name = 'llm-ollama-native'
export const inject = ['llm']

/** @param {import('@deepseek-ai/cordis').Context} ctx @param {{provider: string, baseURL?: string, models: object[]}} config */
export function apply(ctx, config) {
  const adapter = new OllamaNativeAdapter({
    provider: config.provider,
    baseURL: config.baseURL,
    models: config.models,
  })
  ctx.llm.registerAdapter([config.provider], adapter)
}
