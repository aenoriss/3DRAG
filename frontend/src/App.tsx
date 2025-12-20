import { useState, useEffect, useCallback, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

interface Model {
  id: string
  name: string
  category?: string
  file_path?: string
  indexed_at?: string
}

interface ProcessingModel {
  id: string
  name: string
  status: 'processing' | 'done' | 'error'
  images?: string[]
  error?: string
}

interface DatasetStatus {
  is_generating: boolean
  total: number
  downloaded: number
  indexed: number
  failed: number
  current_model?: string
  step?: string
  message?: string
}

function App() {
  const [models, setModels] = useState<Model[]>([])
  const [processingModels, setProcessingModels] = useState<ProcessingModel[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [wsConnected, setWsConnected] = useState(false)
  const [mode, setMode] = useState<string>('unknown')
  const [error, setError] = useState<string | null>(null)
  const [datasetStatus, setDatasetStatus] = useState<DatasetStatus | null>(null)
  const [datasetCount, setDatasetCount] = useState(100)
  const wsRef = useRef<WebSocket | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Fetch models on mount
  useEffect(() => {
    fetchModels()
  }, [])

  // WebSocket connection
  useEffect(() => {
    const connectWS = () => {
      const ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        setWsConnected(true)
        console.log('WebSocket connected')
      }

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data)
        handleWSMessage(data)
      }

      ws.onclose = () => {
        setWsConnected(false)
        console.log('WebSocket disconnected, reconnecting...')
        setTimeout(connectWS, 3000)
      }

      ws.onerror = (err) => {
        console.error('WebSocket error:', err)
      }

      wsRef.current = ws
    }

    connectWS()

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const handleWSMessage = (data: any) => {
    console.log('WS message:', data)

    switch (data.type) {
      case 'connected':
        setMode(data.mode)
        break

      case 'model_processing':
        setProcessingModels(prev => [
          ...prev,
          { id: data.model_id, name: data.name, status: 'processing' }
        ])
        break

      case 'model_added':
        setProcessingModels(prev =>
          prev.map(m =>
            m.id === data.model_id
              ? { ...m, status: 'done', images: data.images_b64 }
              : m
          )
        )
        // Refresh models list
        fetchModels()
        break

      case 'model_error':
        setProcessingModels(prev =>
          prev.map(m =>
            m.id === data.model_id
              ? { ...m, status: 'error', error: data.error }
              : m
          )
        )
        break

      case 'dataset_status':
        setDatasetStatus(prev => ({
          ...prev,
          is_generating: true,
          total: data.total || prev?.total || 0,
          downloaded: data.downloaded || prev?.downloaded || 0,
          indexed: data.indexed || prev?.indexed || 0,
          failed: data.failed || prev?.failed || 0,
          step: data.status || prev?.step,
          message: data.message || prev?.message
        }))
        break

      case 'dataset_progress':
        setDatasetStatus(prev => ({
          ...prev,
          is_generating: true,
          total: data.total ?? prev?.total ?? 0,
          downloaded: data.downloaded ?? prev?.downloaded ?? 0,
          indexed: data.indexed ?? prev?.indexed ?? 0,
          failed: data.failed ?? prev?.failed ?? 0,
          current_model: data.current || data.current_model,
          step: data.step || prev?.step,
          message: data.message || prev?.message
        }))
        break

      case 'dataset_complete':
        setDatasetStatus(null)
        fetchModels()
        break

      case 'dataset_error':
        setDatasetStatus(null)
        setError(`Dataset error: ${data.error}`)
        break

      case 'dataset_cleared':
        fetchModels()
        break

      case 'dataset_cancelled':
        setDatasetStatus(null)
        break
    }
  }

  const fetchModels = async () => {
    try {
      const res = await fetch(`${API_URL}/models`)
      const data = await res.json()
      setModels(data.models || [])
    } catch (err) {
      console.error('Failed to fetch models:', err)
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const files = Array.from(e.dataTransfer.files)
    await uploadFiles(files)
  }, [])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    await uploadFiles(files)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const uploadFiles = async (files: File[]) => {
    const supportedExtensions = ['.glb', '.gltf', '.obj', '.stl', '.ply', '.fbx', '.dae', '.3ds', '.off']

    for (const file of files) {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      if (!supportedExtensions.includes(ext)) {
        setError(`Unsupported format: ${ext}. Supported: ${supportedExtensions.join(', ')}`)
        continue
      }

      const modelId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      const modelName = file.name.replace(/\.[^/.]+$/, '')

      // Add to processing list immediately
      setProcessingModels(prev => [
        ...prev,
        { id: modelId, name: modelName, status: 'processing' }
      ])

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('model_id', modelId)
        formData.append('name', modelName)
        formData.append('include_stats', 'true')

        const res = await fetch(`${API_URL}/models`, {
          method: 'POST',
          body: formData,
        })

        if (!res.ok) {
          const error = await res.json()
          throw new Error(error.detail || 'Upload failed')
        }

        // Model added successfully - WebSocket will handle the rest
      } catch (err: any) {
        setProcessingModels(prev =>
          prev.map(m =>
            m.id === modelId
              ? { ...m, status: 'error', error: err.message }
              : m
          )
        )
      }
    }
  }

  const clearProcessed = () => {
    setProcessingModels(prev => prev.filter(m => m.status === 'processing'))
  }

  const generateDataset = async () => {
    try {
      const res = await fetch(`${API_URL}/dataset/generate?count=${datasetCount}`, {
        method: 'POST'
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Failed to start generation')
      }
      setDatasetStatus({
        is_generating: true,
        total: datasetCount,
        downloaded: 0,
        indexed: 0,
        failed: 0
      })
    } catch (err: any) {
      setError(err.message)
    }
  }

  const deleteDataset = async () => {
    if (!confirm('Delete all indexed models?')) return
    try {
      const res = await fetch(`${API_URL}/dataset`, { method: 'DELETE' })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Failed to delete')
      }
      setModels([])
    } catch (err: any) {
      setError(err.message)
    }
  }

  const cancelGeneration = async () => {
    try {
      await fetch(`${API_URL}/dataset/cancel`, { method: 'POST' })
      setDatasetStatus(null)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const indexExisting = async () => {
    try {
      const res = await fetch(`${API_URL}/dataset/index-existing?limit=${datasetCount}`, {
        method: 'POST'
      })
      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Failed to index')
      }
      setDatasetStatus({
        is_generating: true,
        total: datasetCount,
        downloaded: datasetCount,
        indexed: 0,
        failed: 0,
        step: 'indexing',
        message: 'Indexing cached models...'
      })
    } catch (err: any) {
      setError(err.message)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      {/* Header */}
      <div className="max-w-6xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold">3D Model Search</h1>
            <p className="text-gray-400 mt-1">Drag & drop 3D models to index them</p>
          </div>
          <div className="flex items-center gap-4">
            <span className={`px-3 py-1 rounded-full text-sm ${
              wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {wsConnected ? '● Connected' : '○ Disconnected'}
            </span>
            <span className="px-3 py-1 rounded-full text-sm bg-blue-500/20 text-blue-400">
              Mode: {mode}
            </span>
          </div>
        </div>

        {/* Error Banner */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/20 border border-red-500 rounded-lg flex justify-between items-center">
            <span className="text-red-400">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">✕</button>
          </div>
        )}

        {/* Dataset Generation */}
        <div className="mb-8 p-6 bg-gray-800 rounded-xl border border-gray-700">
          <h2 className="text-lg font-semibold mb-4">Objaverse Dataset</h2>

          {datasetStatus?.is_generating ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
                  <span className="text-yellow-400 font-medium">
                    {datasetStatus.step === 'downloading' ? 'Downloading' :
                     datasetStatus.step === 'indexing' ? 'Indexing' :
                     datasetStatus.step === 'clearing' ? 'Clearing' :
                     'Generating'}
                  </span>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={cancelGeneration}
                    className="px-3 py-1 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded text-sm transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>

              {/* Step indicators */}
              <div className="flex items-center gap-2 text-xs">
                <div className={`flex items-center gap-1 ${datasetStatus.step === 'clearing' ? 'text-yellow-400' : datasetStatus.downloaded > 0 ? 'text-green-400' : 'text-gray-500'}`}>
                  <span>{datasetStatus.downloaded > 0 ? '✓' : '○'}</span>
                  <span>Clear</span>
                </div>
                <span className="text-gray-600">→</span>
                <div className={`flex items-center gap-1 ${datasetStatus.step === 'downloading' ? 'text-yellow-400' : datasetStatus.downloaded > 0 && datasetStatus.step === 'indexing' ? 'text-green-400' : 'text-gray-500'}`}>
                  <span>{datasetStatus.step === 'downloading' ? '◉' : datasetStatus.downloaded > 0 && datasetStatus.step === 'indexing' ? '✓' : '○'}</span>
                  <span>Download ({datasetStatus.downloaded}/{datasetStatus.total})</span>
                </div>
                <span className="text-gray-600">→</span>
                <div className={`flex items-center gap-1 ${datasetStatus.step === 'indexing' ? 'text-yellow-400' : 'text-gray-500'}`}>
                  <span>{datasetStatus.step === 'indexing' ? '◉' : '○'}</span>
                  <span>Index ({datasetStatus.indexed}/{datasetStatus.downloaded || datasetStatus.total})</span>
                </div>
              </div>

              {/* Progress bar */}
              <div className="w-full bg-gray-700 rounded-full h-2">
                <div
                  className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${datasetStatus.step === 'indexing'
                      ? ((datasetStatus.indexed / (datasetStatus.downloaded || datasetStatus.total || 1)) * 100)
                      : datasetStatus.step === 'downloading'
                      ? ((datasetStatus.downloaded / (datasetStatus.total || 1)) * 100)
                      : 0}%`
                  }}
                />
              </div>

              {/* Current status message */}
              {datasetStatus.message && (
                <p className="text-xs text-gray-400">
                  {datasetStatus.message}
                </p>
              )}
              {datasetStatus.current_model && (
                <p className="text-xs text-gray-500 truncate">
                  Current: {datasetStatus.current_model}
                </p>
              )}
              {datasetStatus.failed > 0 && (
                <p className="text-xs text-red-400">
                  Failed: {datasetStatus.failed}
                </p>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <label className="text-sm text-gray-400">Count:</label>
                <input
                  type="number"
                  min={10}
                  max={1000}
                  value={datasetCount}
                  onChange={(e) => setDatasetCount(Number(e.target.value))}
                  className="w-20 px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
                />
              </div>
              <button
                onClick={generateDataset}
                className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm font-medium transition-colors"
              >
                Download & Index
              </button>
              <button
                onClick={indexExisting}
                className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-sm font-medium transition-colors"
              >
                Index Cached
              </button>
              {models.length > 0 && (
                <button
                  onClick={deleteDataset}
                  className="px-4 py-2 bg-red-600/20 hover:bg-red-600/30 text-red-400 rounded-lg text-sm font-medium transition-colors"
                >
                  Delete
                </button>
              )}
            </div>
          )}
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            mb-8 p-12 border-2 border-dashed rounded-xl cursor-pointer
            transition-all duration-200 text-center
            ${isDragging
              ? 'border-blue-400 bg-blue-500/10'
              : 'border-gray-600 hover:border-gray-500 hover:bg-gray-800/50'
            }
          `}
        >
          <div className="text-5xl mb-4">📦</div>
          <p className="text-lg font-medium">
            {isDragging ? 'Drop your 3D models here' : 'Drag & drop 3D models'}
          </p>
          <p className="text-gray-500 mt-2">
            GLB, GLTF, OBJ, STL, PLY, FBX, DAE
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".glb,.gltf,.obj,.stl,.ply,.fbx,.dae,.3ds,.off"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>

        {/* Processing Models */}
        {processingModels.length > 0 && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Processing</h2>
              <button
                onClick={clearProcessed}
                className="text-sm text-gray-400 hover:text-white"
              >
                Clear completed
              </button>
            </div>
            <div className="space-y-4">
              {processingModels.map(model => (
                <div
                  key={model.id}
                  className={`p-4 rounded-lg ${
                    model.status === 'error'
                      ? 'bg-red-500/10 border border-red-500/50'
                      : model.status === 'done'
                      ? 'bg-green-500/10 border border-green-500/50'
                      : 'bg-gray-800 border border-gray-700'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium">{model.name}</span>
                    <span className={`text-sm ${
                      model.status === 'error' ? 'text-red-400' :
                      model.status === 'done' ? 'text-green-400' :
                      'text-yellow-400'
                    }`}>
                      {model.status === 'processing' && '⏳ Processing...'}
                      {model.status === 'done' && '✓ Done'}
                      {model.status === 'error' && `✕ ${model.error}`}
                    </span>
                  </div>

                  {/* Rendered Images Grid */}
                  {model.images && model.images.length > 0 && (
                    <div className="grid grid-cols-6 gap-2 mt-3">
                      {model.images.map((img, i) => (
                        <img
                          key={i}
                          src={`data:image/png;base64,${img}`}
                          alt={`View ${i + 1}`}
                          className="w-full aspect-square rounded bg-gray-700 object-cover"
                        />
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Indexed Models */}
        <div>
          <h2 className="text-xl font-semibold mb-4">
            Indexed Models ({models.length})
          </h2>
          {models.length === 0 ? (
            <div className="text-center py-12 text-gray-500">
              <p>No models indexed yet</p>
              <p className="text-sm mt-1">Drop some 3D models above to get started</p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {models.map(model => (
                <div
                  key={model.id}
                  className="p-4 bg-gray-800 rounded-lg border border-gray-700"
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-medium">{model.name}</h3>
                      {model.category && (
                        <span className="text-sm text-gray-400">{model.category}</span>
                      )}
                    </div>
                    <span className="text-xs text-gray-500 font-mono">
                      {model.id.slice(0, 12)}...
                    </span>
                  </div>
                  {model.file_path && (
                    <p className="text-xs text-gray-500 mt-2 truncate">
                      {model.file_path}
                    </p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
