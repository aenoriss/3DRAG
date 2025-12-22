import { Suspense, useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, useGLTF, Environment, Center, Html } from '@react-three/drei'
import * as THREE from 'three'

interface ModelProps {
  url: string
}

function Model({ url }: ModelProps) {
  const { scene } = useGLTF(url)
  const ref = useRef<THREE.Group>(null)

  // Auto-rotate slowly
  useFrame((_, delta) => {
    if (ref.current) {
      ref.current.rotation.y += delta * 0.3
    }
  })

  return (
    <Center>
      <primitive ref={ref} object={scene} />
    </Center>
  )
}

function Loader() {
  return (
    <Html center>
      <div className="text-white text-lg animate-pulse">Loading 3D model...</div>
    </Html>
  )
}

interface ModelViewerProps {
  modelId: string
  modelName: string
  apiUrl: string
  onClose: () => void
}

export default function ModelViewer({ modelId, modelName, apiUrl, onClose }: ModelViewerProps) {
  const modelUrl = `${apiUrl}/storage/model/${modelId}`

  return (
    <div
      className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div
        className="bg-gray-800 rounded-xl w-[90vw] h-[85vh] max-w-5xl flex flex-col overflow-hidden border border-gray-700"
        onClick={e => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div>
            <h2 className="text-xl font-semibold text-white">{modelName}</h2>
            <p className="text-sm text-gray-400">Click and drag to rotate, scroll to zoom</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-700 rounded-lg text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 3D Viewer */}
        <div className="flex-1 bg-gray-900">
          <Canvas
            camera={{ position: [0, 0, 5], fov: 50 }}
            gl={{ antialias: true, alpha: true }}
          >
            <ambientLight intensity={0.5} />
            <directionalLight position={[10, 10, 5]} intensity={1} />
            <directionalLight position={[-10, -10, -5]} intensity={0.5} />

            <Suspense fallback={<Loader />}>
              <Model url={modelUrl} />
              <Environment preset="city" />
            </Suspense>

            <OrbitControls
              enablePan={true}
              enableZoom={true}
              enableRotate={true}
              autoRotate={false}
            />
          </Canvas>
        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-gray-700 flex justify-between items-center">
          <span className="text-sm text-gray-500">ID: {modelId}</span>
          <a
            href={modelUrl}
            download
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition-colors"
          >
            Download GLB
          </a>
        </div>
      </div>
    </div>
  )
}
