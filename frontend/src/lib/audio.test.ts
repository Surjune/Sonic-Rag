/**
 * Silence-trimming checks.
 *
 * Run with `npx tsx src/lib/audio.test.ts`. Kept dependency-free rather than
 * pulling a test runner into the frontend for one pure function.
 */

import { encodeWav, trimSilence } from './audio'

let failures = 0

function check(name: string, condition: boolean): void {
  if (condition) {
    console.log(`  ok   ${name}`)
  } else {
    console.error(`  FAIL ${name}`)
    failures += 1
  }
}

function withSpeech(leading: number, speech: number, trailing: number): Float32Array {
  const samples = new Float32Array(leading + speech + trailing)
  for (let i = leading; i < leading + speech; i += 1) {
    samples[i] = 0.5 * Math.sin(i / 4)
  }
  return samples
}

console.log('trimSilence')

// 1s of silence, 1s of speech, 1s of silence, at 16kHz.
const padded = withSpeech(16000, 16000, 16000)
const trimmed = trimSilence(padded)
check('removes leading and trailing silence', trimmed.length < padded.length)
check('keeps the speech region', trimmed.length >= 16000)

// Padding must survive so a quiet word onset is not clipped.
check('retains padding around speech', trimmed.length > 16000)

const allSilent = new Float32Array(8000)
check(
  'silent input is returned unchanged, not emptied',
  trimSilence(allSilent).length === allSilent.length,
)

const allSpeech = withSpeech(0, 8000, 0)
check('speech-only input is preserved', trimSilence(allSpeech).length === allSpeech.length)

check('empty input does not throw', trimSilence(new Float32Array(0)).length === 0)

console.log('encodeWav')
const wav = encodeWav(new Float32Array([0, 0.5, -0.5, 1, -1]), 16000)
check('produces a wav blob', wav.type === 'audio/wav')
check('header plus 2 bytes per sample', wav.size === 44 + 5 * 2)

if (failures > 0) {
  console.error(`\n${failures} check(s) failed`)
  process.exit(1)
}
console.log('\nall checks passed')
