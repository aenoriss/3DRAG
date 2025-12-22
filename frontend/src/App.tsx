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
  current_images?: string[]
}

interface SearchResult {
  id: string
  name: string
  score: number
  distance: number
  category?: string
  file_path?: string
}

interface CumulativeStats {
  total_requests: number
  total_embeddings: number
  total_text_queries: number
  total_time_sec: number
  total_vision_tokens: number
  avg_time_sec: number
  uptime_sec: number
  estimated_cost_usd: number
  cost_per_model_usd: number
  gpu_cost_per_sec: number
}

interface OllamaStats {
  status: string
  mode: string
  endpoint?: string
  models?: string[]
  vision_model?: string
  embedding_model?: string
  embedding_dim?: number
  cumulative?: CumulativeStats
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
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [ollamaStats, setOllamaStats] = useState<OllamaStats | null>(null)
  const [isLoadingStats, setIsLoadingStats] = useState(false)
  const [showStats, setShowStats] = useState(false)
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
          message: data.message || prev?.message,
          current_images: data.images || prev?.current_images
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

      case 'batch_start':
        setProcessingModels([{
          id: 'batch',
          name: `Processing ${data.total} models...`,
          status: 'processing'
        }])
        break

      case 'batch_complete':
        setProcessingModels([{
          id: 'batch',
          name: `Processed ${data.added}/${data.total} models in ${data.time_sec?.toFixed(1)}s`,
          status: data.failed > 0 ? 'error' : 'done',
          error: data.failed > 0 ? `${data.failed} failed` : undefined
        }])
        fetchModels()
        break

      case 'batch_error':
        setProcessingModels([{
          id: 'batch',
          name: 'Batch processing failed',
          status: 'error',
          error: data.error
        }])
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

  const fetchStats = async () => {
    setIsLoadingStats(true)
    try {
      const res = await fetch(`${API_URL}/stats?include_backend=true`)
      const data = await res.json()
      if (data.ollama) {
        setOllamaStats(data.ollama)
      }
    } catch (err) {
      console.error('Failed to fetch stats:', err)
    } finally {
      setIsLoadingStats(false)
    }
  }

  const resetStats = async () => {
    try {
      await fetch(`${API_URL}/stats/reset`, { method: 'POST' })
      await fetchStats()
    } catch (err: any) {
      setError(`Failed to reset stats: ${err.message}`)
    }
  }

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!searchQuery.trim()) return

    setIsSearching(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(searchQuery)}&k=10`)
      const data = await res.json()
      setSearchResults(data.results || [])
    } catch (err: any) {
      setError(`Search failed: ${err.message}`)
      setSearchResults([])
    } finally {
      setIsSearching(false)
    }
  }

  const clearSearch = () => {
    setSearchQuery('')
    setSearchResults([])
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

    // Check if items contain directories
    const items = Array.from(e.dataTransfer.items)
    const files: File[] = []

    // Helper to recursively get files from directory
    const getFilesFromEntry = async (entry: FileSystemEntry): Promise<File[]> => {
      if (entry.isFile) {
        return new Promise((resolve) => {
          (entry as FileSystemFileEntry).file((file) => resolve([file]))
        })
      } else if (entry.isDirectory) {
        const dirReader = (entry as FileSystemDirectoryEntry).createReader()
        return new Promise((resolve) => {
          const allFiles: File[] = []
          const readEntries = () => {
            dirReader.readEntries(async (entries) => {
              if (entries.length === 0) {
                resolve(allFiles)
              } else {
                for (const entry of entries) {
                  const entryFiles = await getFilesFromEntry(entry)
                  allFiles.push(...entryFiles)
                }
                readEntries() // Continue reading (directories can have >100 entries)
              }
            })
          }
          readEntries()
        })
      }
      return []
    }

    // Process each dropped item
    for (const item of items) {
      const entry = item.webkitGetAsEntry?.()
      if (entry) {
        const entryFiles = await getFilesFromEntry(entry)
        files.push(...entryFiles)
      } else if (item.kind === 'file') {
        const file = item.getAsFile()
        if (file) files.push(file)
      }
    }

    if (files.length > 0) {
      console.log(`Processing ${files.length} files from drop`)
      await uploadFiles(files)
    }
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

    // Filter valid files
    const validFiles = files.filter(file => {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase()
      return supportedExtensions.includes(ext)
    })

    if (validFiles.length === 0) {
      setError(`No valid 3D files found. Supported: ${supportedExtensions.join(', ')}`)
      return
    }

    // Use batch upload for multiple files (>1)
    if (validFiles.length > 1) {
      await uploadBatch(validFiles)
      return
    }

    // Single file upload (original logic)
    const file = validFiles[0]
    const modelId = `model_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const modelName = file.name.replace(/\.[^/.]+$/, '')

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

  const uploadBatch = async (files: File[]) => {
    // Show batch processing state
    setProcessingModels([{
      id: 'batch',
      name: `Processing ${files.length} models...`,
      status: 'processing'
    }])

    try {
      const formData = new FormData()
      files.forEach(file => {
        formData.append('files', file)
      })

      console.log(`[batch] Uploading ${files.length} files...`)

      const res = await fetch(`${API_URL}/models/batch?clear=true`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const error = await res.json()
        throw new Error(error.detail || 'Batch upload failed')
      }

      const result = await res.json()
      console.log('[batch] Result:', result)

      setProcessingModels([{
        id: 'batch',
        name: `Processed ${result.added}/${result.total} models in ${result.time_sec?.toFixed(1)}s`,
        status: result.failed > 0 ? 'error' : 'done',
        error: result.failed > 0 ? `${result.failed} failed` : undefined
      }])

      // Refresh models list
      fetchModels()

    } catch (err: any) {
      console.error('[batch] Error:', err)
      setProcessingModels([{
        id: 'batch',
        name: 'Batch upload failed',
        status: 'error',
        error: err.message
      }])
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

        {/* Search */}
        <div className="mb-8 p-6 bg-gray-800 rounded-xl border border-gray-700">
          <h2 className="text-lg font-semibold mb-4">Semantic Search</h2>
          <form onSubmit={handleSearch} className="flex gap-3">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for 3D models... (e.g., 'wooden chair', 'red car')"
              className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500"
            />
            <button
              type="submit"
              disabled={isSearching || !searchQuery.trim()}
              className="px-6 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
            {searchResults.length > 0 && (
              <button
                type="button"
                onClick={clearSearch}
                className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
              >
                Clear
              </button>
            )}
          </form>

          {/* Stats Toggle */}
          <div className="mt-4 flex justify-end">
            <button
              onClick={() => {
                setShowStats(!showStats)
                if (!showStats && !ollamaStats) fetchStats()
              }}
              className="text-sm text-gray-400 hover:text-white flex items-center gap-1"
            >
              {showStats ? '▼' : '▶'} RunPod Metrics
            </button>
          </div>

          {/* Stats Panel */}
          {showStats && (
            <div className="mt-4 p-4 bg-gray-700/50 rounded-lg border border-gray-600">
              <div className="flex items-center justify-between mb-4">
                <h3 className="font-semibold text-gray-200">Embedding Pipeline Stats</h3>
                <div className="flex gap-2">
                  <button
                    onClick={fetchStats}
                    disabled={isLoadingStats}
                    className="px-3 py-1 text-xs bg-gray-600 hover:bg-gray-500 rounded transition-colors disabled:opacity-50"
                  >
                    {isLoadingStats ? 'Loading...' : 'Refresh'}
                  </button>
                  <button
                    onClick={resetStats}
                    className="px-3 py-1 text-xs bg-red-600/30 hover:bg-red-600/50 text-red-400 rounded transition-colors"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {ollamaStats ? (
                <div className="space-y-4">
                  {/* Status Row */}
                  <div className="flex items-center gap-4 text-sm">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      ollamaStats.status === 'ok' || ollamaStats.status === 'ready'
                        ? 'bg-green-500/20 text-green-400'
                        : 'bg-red-500/20 text-red-400'
                    }`}>
                      {ollamaStats.status === 'ok' || ollamaStats.status === 'ready' ? '● Online' : '○ Offline'}
                    </span>
                    <span className="text-gray-400">
                      {ollamaStats.vision_model} + {ollamaStats.embedding_model}
                    </span>
                    <span className="text-gray-500 text-xs">
                      {ollamaStats.embedding_dim}-dim
                    </span>
                  </div>

                  {/* Metrics Grid */}
                  {ollamaStats.cumulative && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      {/* Total Embeddings */}
                      <div className="p-3 bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-blue-400">
                          {ollamaStats.cumulative.total_embeddings}
                        </div>
                        <div className="text-xs text-gray-400">Models Indexed</div>
                      </div>

                      {/* Avg Time */}
                      <div className="p-3 bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-green-400">
                          {ollamaStats.cumulative.avg_time_sec.toFixed(2)}s
                        </div>
                        <div className="text-xs text-gray-400">Avg Time/Model</div>
                      </div>

                      {/* Total Cost */}
                      <div className="p-3 bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-yellow-400">
                          ${ollamaStats.cumulative.estimated_cost_usd.toFixed(4)}
                        </div>
                        <div className="text-xs text-gray-400">Total GPU Cost</div>
                      </div>

                      {/* Cost per Model */}
                      <div className="p-3 bg-gray-800 rounded-lg">
                        <div className="text-2xl font-bold text-purple-400">
                          ${ollamaStats.cumulative.cost_per_model_usd.toFixed(5)}
                        </div>
                        <div className="text-xs text-gray-400">Cost/Model</div>
                      </div>
                    </div>
                  )}

                  {/* Secondary Stats */}
                  {ollamaStats.cumulative && (
                    <div className="flex flex-wrap gap-4 text-xs text-gray-400 pt-2 border-t border-gray-600">
                      <span>Total Time: {ollamaStats.cumulative.total_time_sec.toFixed(1)}s</span>
                      <span>Vision Tokens: {ollamaStats.cumulative.total_vision_tokens.toLocaleString()}</span>
                      <span>Text Queries: {ollamaStats.cumulative.total_text_queries}</span>
                      <span>Uptime: {Math.floor(ollamaStats.cumulative.uptime_sec / 60)}m</span>
                      <span>GPU Rate: ${ollamaStats.cumulative.gpu_cost_per_sec}/sec</span>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-center text-gray-400 py-4">
                  {isLoadingStats ? 'Loading stats...' : 'Click Refresh to load stats'}
                </div>
              )}
            </div>
          )}

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="mt-4">
              <p className="text-sm text-gray-400 mb-3">Found {searchResults.length} results for "{searchQuery}"</p>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {searchResults.map((result, i) => (
                  <div
                    key={result.id}
                    className="p-3 bg-gray-700/50 rounded-lg border border-gray-600"
                  >
                    <div className="relative mb-2">
                      <img
                        src={`${API_URL}/previews/${result.id}.jpg`}
                        alt={result.name}
                        className="w-full aspect-square rounded bg-gray-600 object-cover"
                        onError={(e) => {
                          (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect fill="%23444" width="100" height="100"/><text x="50" y="55" text-anchor="middle" fill="%23666" font-size="12">No preview</text></svg>'
                        }}
                      />
                      <span className="absolute top-1 left-1 bg-black/60 text-white text-xs px-1.5 py-0.5 rounded">
                        #{i + 1}
                      </span>
                      <span className="absolute top-1 right-1 bg-blue-600/80 text-white text-xs px-1.5 py-0.5 rounded">
                        {(result.score * 100).toFixed(0)}%
                      </span>
                    </div>
                    <h4 className="font-medium text-sm truncate" title={result.name}>{result.name}</h4>
                    {result.category && (
                      <span className="text-xs text-gray-400">{result.category}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

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

              {/* Rendered Images Preview (4 thumbnail views) */}
              {datasetStatus.current_images && datasetStatus.current_images.length > 0 && (
                <div className="mt-4">
                  <p className="text-xs text-gray-400 mb-2">Rendered Views:</p>
                  <div className="flex gap-2">
                    {datasetStatus.current_images.map((img, i) => (
                      <img
                        key={i}
                        src={`data:image/jpeg;base64,${img}`}
                        alt={`View ${i + 1}`}
                        className="w-16 h-16 rounded bg-gray-700 object-cover"
                      />
                    ))}
                  </div>
                </div>
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
            {isDragging ? 'Drop your 3D models here' : 'Drag & drop 3D models or folders'}
          </p>
          <p className="text-gray-500 mt-2">
            GLB, GLTF, OBJ, STL, PLY, FBX, DAE • Supports folder drops
          </p>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".glb,.gltf,.obj,.stl,.ply,.fbx,.dae,.3ds,.off"
            onChange={handleFileSelect}
            className="hidden"
            {...{ webkitdirectory: "", directory: "" } as any}
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
