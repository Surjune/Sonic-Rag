/**
 * Audio-reactive hologram.
 *
 * An icosphere whose vertices are displaced by 3D simplex noise, with the
 * displacement amplitude driven by live microphone level. Colour encodes the
 * pipeline state, so the visualizer reports what the system is actually doing
 * rather than animating decoratively.
 *
 * Displacement happens in the vertex shader: moving ~10k vertices per frame on
 * the CPU would cost more than the entire retrieval budget it is visualizing.
 */

import { useFrame } from '@react-three/fiber'
import { useMemo, useRef } from 'react'
import * as THREE from 'three'
import type { PipelineState } from '../lib/types'

/** Colour per pipeline state. */
export const STATE_COLORS: Record<PipelineState, string> = {
  idle: '#22d3ee', // cyan
  listening: '#c026d3', // violet/magenta
  processing: '#f59e0b', // amber
  grounded: '#10b981', // emerald
  refused: '#e11d48', // ruby
  error: '#e11d48',
}

const VERTEX_SHADER = /* glsl */ `
  uniform float uTime;
  uniform float uLevel;
  uniform float uSpin;
  varying float vDisplacement;
  varying vec3 vNormal;

  // Ashima simplex noise (3D), public domain.
  vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
  vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
  vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

  float snoise(vec3 v) {
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
              i.z + vec4(0.0, i1.z, i2.z, 1.0))
            + i.y + vec4(0.0, i1.y, i2.y, 1.0))
            + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
  }

  void main() {
    vNormal = normal;
    // Two octaves: a slow swell plus a finer ripple that only shows when loud.
    float slow = snoise(position * 1.2 + uTime * 0.25);
    float fine = snoise(position * 3.5 + uTime * (0.6 + uSpin * 2.0));
    float amount = slow * (0.12 + uLevel * 0.55) + fine * uLevel * 0.22;
    vDisplacement = amount;
    vec3 displaced = position + normal * amount;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
  }
`

const FRAGMENT_SHADER = /* glsl */ `
  uniform vec3 uColor;
  uniform float uOpacity;
  varying float vDisplacement;
  varying vec3 vNormal;

  void main() {
    // Fresnel rim so the sphere reads as a hologram shell, not a solid ball.
    float rim = pow(1.0 - abs(dot(normalize(vNormal), vec3(0.0, 0.0, 1.0))), 2.0);
    float heat = clamp(vDisplacement * 2.0 + 0.35, 0.0, 1.0);
    vec3 color = uColor * (0.45 + heat * 0.9) + vec3(rim * 0.5);
    gl_FragColor = vec4(color, uOpacity * (0.35 + rim * 0.65));
  }
`

interface OrbProps {
  state: PipelineState
  /** Live microphone level in [0,1]. */
  level: number
}

function Hologram({ state, level }: OrbProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  const materialRef = useRef<THREE.ShaderMaterial>(null)
  const smoothedLevel = useRef(0)
  const color = useRef(new THREE.Color(STATE_COLORS.idle))

  const uniforms = useMemo(
    () => ({
      uTime: { value: 0 },
      uLevel: { value: 0 },
      uSpin: { value: 0 },
      uColor: { value: new THREE.Color(STATE_COLORS.idle) },
      uOpacity: { value: 0.9 },
    }),
    [],
  )

  useFrame((_, delta) => {
    const material = materialRef.current
    const mesh = meshRef.current
    if (!material || !mesh) return

    material.uniforms.uTime.value += delta

    // Idle breathes gently; processing spins hard. Without a floor the orb
    // freezes whenever the mic is silent, which reads as "broken".
    const floor = state === 'processing' ? 0.55 : state === 'idle' ? 0.06 : 0.12
    const target = Math.max(level, floor)
    smoothedLevel.current += (target - smoothedLevel.current) * Math.min(1, delta * 8)
    material.uniforms.uLevel.value = smoothedLevel.current
    material.uniforms.uSpin.value = state === 'processing' ? 1 : 0

    // Ease between state colours so transitions do not snap.
    color.current.lerp(new THREE.Color(STATE_COLORS[state]), Math.min(1, delta * 4))
    ;(material.uniforms.uColor.value as THREE.Color).copy(color.current)

    const spin = state === 'processing' ? 1.6 : 0.22
    mesh.rotation.y += delta * spin
    mesh.rotation.x += delta * spin * 0.25
  })

  return (
    <mesh ref={meshRef}>
      <icosahedronGeometry args={[1.35, 32]} />
      <shaderMaterial
        ref={materialRef}
        uniforms={uniforms}
        vertexShader={VERTEX_SHADER}
        fragmentShader={FRAGMENT_SHADER}
        transparent
        wireframe
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </mesh>
  )
}

/** Particle halo that drifts outward with audio level. */
function Halo({ state, level }: OrbProps) {
  const pointsRef = useRef<THREE.Points>(null)
  const materialRef = useRef<THREE.PointsMaterial>(null)
  const color = useRef(new THREE.Color(STATE_COLORS.idle))

  const geometry = useMemo(() => {
    const count = 1400
    const positions = new Float32Array(count * 3)
    for (let i = 0; i < count; i += 1) {
      // Even distribution on a shell, jittered so it does not look banded.
      const theta = Math.acos(2 * Math.random() - 1)
      const phi = Math.random() * Math.PI * 2
      const radius = 1.75 + Math.random() * 0.85
      positions[i * 3] = radius * Math.sin(theta) * Math.cos(phi)
      positions[i * 3 + 1] = radius * Math.sin(theta) * Math.sin(phi)
      positions[i * 3 + 2] = radius * Math.cos(theta)
    }
    const buffer = new THREE.BufferGeometry()
    buffer.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    return buffer
  }, [])

  useFrame((_, delta) => {
    const points = pointsRef.current
    const material = materialRef.current
    if (!points || !material) return
    points.rotation.y -= delta * (state === 'processing' ? 0.75 : 0.1)
    const scale = 1 + level * 0.22
    points.scale.setScalar(scale)
    color.current.lerp(new THREE.Color(STATE_COLORS[state]), Math.min(1, delta * 4))
    material.color.copy(color.current)
    material.opacity = 0.28 + level * 0.5
  })

  return (
    <points ref={pointsRef} geometry={geometry}>
      <pointsMaterial
        ref={materialRef}
        size={0.022}
        transparent
        opacity={0.35}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </points>
  )
}

export function Orb({ state, level }: OrbProps) {
  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[4, 4, 4]} intensity={1.1} />
      <Hologram state={state} level={level} />
      <Halo state={state} level={level} />
    </>
  )
}
