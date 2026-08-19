/**
 * Microphone capture, live frequency analysis, and WAV encoding.
 *
 * MediaRecorder would be simpler but produces webm/opus, which the speech
 * upstream does not accept. Converting server-side would mean an ffmpeg binary
 * plus a transcode on the request path -- latency and a dependency that free
 * hosting tiers do not reliably provide. So we capture raw PCM and encode a
 * WAV in the browser, where the cost is negligible and the format is exact.
 *
 * The AudioContext is opened at the target sample rate, so the browser's own
 * resampler does the downsampling for free.
 */

/** Speech models expect 16kHz mono; higher rates add bytes, not accuracy. */
const TARGET_SAMPLE_RATE = 16000
const FFT_SIZE = 256

export interface RecorderHandle {
  /** Live frequency magnitudes in [0,1], newest frame. */
  getFrequencies: () => Uint8Array
  /** Overall loudness in [0,1], for driving displacement. */
  getLevel: () => number
  /** Stop capture and return the encoded WAV. */
  stop: () => Promise<Blob>
  /** Abort without producing audio. */
  cancel: () => void
}

/** Encode float PCM samples as a 16-bit mono WAV. */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const writeString = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i))
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM header size
  view.setUint16(20, 1, true) // format: PCM
  view.setUint16(22, 1, true) // channels: mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let offset = 44
  for (let i = 0; i < samples.length; i += 1) {
    // Clamp before scaling: values outside [-1,1] would wrap and click.
    const clamped = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
    offset += 2
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

/** The worklet runs on the audio thread and forwards raw PCM frames. */
const WORKLET_SOURCE = `
class PcmCollector extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel && channel.length) {
      this.port.postMessage(new Float32Array(channel))
    }
    return true
  }
}
registerProcessor('pcm-collector', PcmCollector)
`

export async function startRecording(): Promise<RecorderHandle> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  })

  const context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE })
  const source = context.createMediaStreamSource(stream)

  const analyser = context.createAnalyser()
  analyser.fftSize = FFT_SIZE
  analyser.smoothingTimeConstant = 0.75
  source.connect(analyser)

  const frequencies = new Uint8Array(analyser.frequencyBinCount)
  const chunks: Float32Array[] = []

  const workletUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: 'text/javascript' }))
  await context.audioWorklet.addModule(workletUrl)
  URL.revokeObjectURL(workletUrl)

  const collector = new AudioWorkletNode(context, 'pcm-collector')
  collector.port.onmessage = (event: MessageEvent<Float32Array>) => {
    chunks.push(event.data)
  }
  source.connect(collector)
  // The worklet emits no audio, but Chrome only pulls from a node that is
  // connected to the graph's destination.
  const silence = context.createGain()
  silence.gain.value = 0
  collector.connect(silence).connect(context.destination)

  const teardown = () => {
    collector.port.onmessage = null
    collector.disconnect()
    source.disconnect()
    stream.getTracks().forEach((track) => track.stop())
    void context.close()
  }

  return {
    getFrequencies: () => {
      analyser.getByteFrequencyData(frequencies)
      return frequencies
    },
    getLevel: () => {
      analyser.getByteFrequencyData(frequencies)
      let sum = 0
      for (let i = 0; i < frequencies.length; i += 1) sum += frequencies[i]
      return sum / frequencies.length / 255
    },
    stop: async () => {
      const sampleRate = context.sampleRate
      teardown()
      const total = chunks.reduce((count, chunk) => count + chunk.length, 0)
      const merged = new Float32Array(total)
      let offset = 0
      for (const chunk of chunks) {
        merged.set(chunk, offset)
        offset += chunk.length
      }
      return encodeWav(merged, sampleRate)
    },
    cancel: teardown,
  }
}
