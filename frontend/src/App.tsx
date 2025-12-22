import { useState, useEffect, useRef } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws'

interface BucketInfo {
  name: string
  region: string
  endpoint: string
  url: string
}

interface BucketFile {
  key: string
  name: string
  size: number
  extension: string
  url: string
  last_modified: string
}

interface ProcessingState {
  isProcessing: boolean
  batch: number
  totalBatches: number
  processed: number
  failed: number
  total: number
  startTime?: number
}

interface SearchResult {
  id: string
  name: string
  score: number
  distance: number
  category?: string
  caption?: string
}

interface Model {
  id: string
  name: string
  category?: string
  caption?: string
}

function App() {
  // Bucket state
  const [bucket, setBucket] = useState<BucketInfo | null>(null)
  const [folders, setFolders] = useState<string[]>([])
  const [currentPrefix, setCurrentPrefix] = useState('')
  const [files, setFiles] = useState<BucketFile[]>([])
  const [isLoadingFiles, setIsLoadingFiles] = useState(false)
  const [isLoadingFolders, setIsLoadingFolders] = useState(false)
  const [folderDropdownOpen, setFolderDropdownOpen] = useState(false)
  const [loadSuccess, setLoadSuccess] = useState<string | null>(null)

  // Processing state
  const [processing, setProcessing] = useState<ProcessingState>({
    isProcessing: false,
    batch: 0,
    totalBatches: 0,
    processed: 0,
    failed: 0,
    total: 0
  })
  const [batchSize, setBatchSize] = useState(100)
  const [clearIndex, setClearIndex] = useState(true)

  // Search state
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)

  // Index state
  const [models, setModels] = useState<Model[]>([])
  const [wsConnected, setWsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [completionStats, setCompletionStats] = useState<{added: number, failed: number, time: number} | null>(null)

  const wsRef = useRef<WebSocket | null>(null)

  // Fetch bucket info on mount
  useEffect(() => {
    fetchBucket()
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
        setTimeout(connectWS, 3000)
      }

      ws.onerror = (err) => console.error('WebSocket error:', err)
      wsRef.current = ws
    }

    connectWS()
    return () => wsRef.current?.close()
  }, [])

  const handleWSMessage = (data: any) => {
    console.log('WS:', data)

    switch (data.type) {
      case 'bucket_process_start':
        setProcessing({
          isProcessing: true,
          batch: 0,
          totalBatches: Math.ceil(data.total / (data.batch_size || 100)),
          processed: 0,
          failed: 0,
          total: data.total,
          startTime: Date.now()
        })
        setCompletionStats(null)
        break

      case 'bucket_process_progress':
        setProcessing(prev => ({
          ...prev,
          batch: data.batch,
          totalBatches: data.total_batches,
          processed: data.processed,
          failed: data.failed,
          total: data.total
        }))
        break

      case 'bucket_process_complete':
        setProcessing(prev => ({ ...prev, isProcessing: false }))
        setCompletionStats({
          added: data.added,
          failed: data.failed,
          time: data.time_sec
        })
        fetchModels()
        break
    }
  }

  const fetchBucket = async () => {
    try {
      const res = await fetch(`${API_URL}/storage/bucket`)
      const data = await res.json()
      setBucket(data)
      fetchFolders('')
    } catch (err) {
      console.error('Failed to fetch bucket:', err)
    }
  }

  const fetchFolders = async (prefix: string) => {
    setIsLoadingFolders(true)
    try {
      const res = await fetch(`${API_URL}/storage/folders?prefix=${encodeURIComponent(prefix)}`)
      const data = await res.json()
      setFolders(data.folders || [])
    } catch (err) {
      console.error('Failed to fetch folders:', err)
      setError('Failed to load folders')
    } finally {
      setIsLoadingFolders(false)
    }
  }

  const fetchFiles = async (prefix: string) => {
    setIsLoadingFiles(true)
    setLoadSuccess(null)
    try {
      const res = await fetch(`${API_URL}/storage/files?prefix=${encodeURIComponent(prefix)}&limit=5000`)
      const data = await res.json()
      setFiles(data.files || [])
      if (data.files?.length > 0) {
        setLoadSuccess(`Loaded ${data.files.length} models from ${prefix || 'root'}`)
        setTimeout(() => setLoadSuccess(null), 3000)
      }
    } catch (err) {
      console.error('Failed to fetch files:', err)
      setFiles([])
      setError('Failed to load files')
    } finally {
      setIsLoadingFiles(false)
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

  const selectFolder = (folder: string) => {
    const newPrefix = currentPrefix ? `${currentPrefix}/${folder}` : folder
    setCurrentPrefix(newPrefix)
    fetchFolders(newPrefix)
    fetchFiles(newPrefix)
  }

  const goBack = () => {
    const parts = currentPrefix.split('/').filter(Boolean)
    parts.pop()
    const newPrefix = parts.join('/')
    setCurrentPrefix(newPrefix)
    fetchFolders(newPrefix)
    if (newPrefix) {
      fetchFiles(newPrefix)
    } else {
      setFiles([])
    }
  }

  const processFiles = async () => {
    if (files.length === 0) return

    setError(null)
    try {
      const res = await fetch(
        `${API_URL}/storage/process?prefix=${encodeURIComponent(currentPrefix)}&clear=${clearIndex}&batch_size=${batchSize}`,
        { method: 'POST' }
      )
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Processing failed')
      }
    } catch (err: any) {
      setError(err.message)
      setProcessing(prev => ({ ...prev, isProcessing: false }))
    }
  }

  const handleSearch = async (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!searchQuery.trim()) return

    setIsSearching(true)
    setError(null)
    try {
      const res = await fetch(`${API_URL}/search?q=${encodeURIComponent(searchQuery)}&k=12`)
      const data = await res.json()
      setSearchResults(data.results || [])
    } catch (err: any) {
      setError(`Search failed: ${err.message}`)
    } finally {
      setIsSearching(false)
    }
  }

  const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}m ${secs.toFixed(0)}s`
  }

  const progressPercent = processing.total > 0
    ? ((processing.processed + processing.failed) / processing.total) * 100
    : 0

  const elapsed = processing.startTime
    ? (Date.now() - processing.startTime) / 1000
    : 0

  const eta = processing.processed > 0 && elapsed > 0
    ? (elapsed / processing.processed) * (processing.total - processing.processed - processing.failed)
    : 0

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">3D Model Search</h1>
            <p className="text-gray-400 text-sm">Production Pipeline</p>
          </div>
          <div className="flex items-center gap-4">
            <span className="text-sm text-gray-400">
              {models.length} models indexed
            </span>
            <span className={`px-3 py-1 rounded-full text-xs ${
              wsConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
            }`}>
              {wsConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto p-6 space-y-6">
        {/* Success Message */}
        {loadSuccess && (
          <div className="p-4 bg-green-500/20 border border-green-500/50 rounded-lg flex justify-between items-center animate-pulse">
            <span className="text-green-400">{loadSuccess}</span>
          </div>
        )}

        {/* Error Banner */}
        {error && (
          <div className="p-4 bg-red-500/20 border border-red-500/50 rounded-lg flex justify-between items-center">
            <span className="text-red-400">{error}</span>
            <button onClick={() => setError(null)} className="text-red-400 hover:text-red-300">×</button>
          </div>
        )}

        {/* Search Section */}
        <section className="bg-gray-800 rounded-xl border border-gray-700 p-6">
          <h2 className="text-lg font-semibold mb-4">Semantic Search</h2>
          <form onSubmit={handleSearch} className="flex gap-3">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search 3D models... (e.g., 'wooden chair', 'red sports car')"
              className="flex-1 px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg focus:outline-none focus:border-blue-500 text-lg"
            />
            <button
              type="submit"
              disabled={isSearching || !searchQuery.trim()}
              className="px-8 py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
            >
              {isSearching ? 'Searching...' : 'Search'}
            </button>
          </form>

          {/* Search Results */}
          {searchResults.length > 0 && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-4">
                <p className="text-gray-400">Found {searchResults.length} results</p>
                <button
                  onClick={() => setSearchResults([])}
                  className="text-sm text-gray-400 hover:text-white"
                >
                  Clear
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                {searchResults.map((result, i) => (
                  <div key={result.id} className="bg-gray-700/50 rounded-lg overflow-hidden border border-gray-600">
                    <div className="relative aspect-square">
                      <img
                        src={`${API_URL}/previews/${result.id}.jpg`}
                        alt={result.name}
                        className="w-full h-full object-cover"
                        onError={(e) => {
                          (e.target as HTMLImageElement).style.display = 'none'
                        }}
                      />
                      <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                        #{i + 1}
                      </div>
                      <div className="absolute top-2 right-2 bg-blue-600/90 text-white text-xs px-2 py-1 rounded font-medium">
                        {(result.score * 100).toFixed(0)}%
                      </div>
                    </div>
                    <div className="p-3">
                      <h4 className="font-medium text-sm truncate" title={result.name}>{result.name}</h4>
                      {result.caption && (
                        <p className="text-xs text-gray-400 mt-1 line-clamp-2">{result.caption}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Bucket Browser Section */}
        <section className="bg-gray-800 rounded-xl border border-gray-700 p-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-lg font-semibold">Bucket Storage</h2>
              {bucket && (
                <p className="text-gray-400 text-sm">{bucket.name} ({bucket.region})</p>
              )}
            </div>
          </div>

          {/* Breadcrumb / Path */}
          <div className="flex items-center gap-2 mb-4 text-sm">
            <button
              onClick={() => { setCurrentPrefix(''); fetchFolders(''); setFiles([]) }}
              className="text-blue-400 hover:text-blue-300"
            >
              {bucket?.name || 'root'}
            </button>
            {currentPrefix.split('/').filter(Boolean).map((part, i, arr) => (
              <span key={i} className="flex items-center gap-2">
                <span className="text-gray-500">/</span>
                <button
                  onClick={() => {
                    const newPrefix = arr.slice(0, i + 1).join('/')
                    setCurrentPrefix(newPrefix)
                    fetchFolders(newPrefix)
                    fetchFiles(newPrefix)
                  }}
                  className="text-blue-400 hover:text-blue-300"
                >
                  {part}
                </button>
              </span>
            ))}
          </div>

          {/* Folder Dropdown */}
          {folders.length > 0 && (
            <div className="mb-4 relative">
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Select Folder</p>
              <div className="flex items-center gap-2">
                {currentPrefix && (
                  <button
                    onClick={goBack}
                    disabled={isLoadingFiles || isLoadingFolders}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm flex items-center gap-2 transition-colors"
                  >
                    <span>←</span> Back
                  </button>
                )}
                <div className="relative flex-1 max-w-md">
                  <button
                    onClick={() => setFolderDropdownOpen(!folderDropdownOpen)}
                    disabled={isLoadingFiles || isLoadingFolders}
                    className="w-full px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm flex items-center justify-between gap-2 transition-colors border border-gray-600"
                  >
                    <span className="flex items-center gap-2">
                      {isLoadingFolders ? (
                        <span className="animate-spin">⏳</span>
                      ) : (
                        <span className="text-yellow-400">📁</span>
                      )}
                      {isLoadingFolders ? 'Loading...' : 'Select a folder'}
                    </span>
                    <span className={`transition-transform ${folderDropdownOpen ? 'rotate-180' : ''}`}>▼</span>
                  </button>
                  {folderDropdownOpen && !isLoadingFolders && (
                    <div className="absolute top-full left-0 right-0 mt-1 bg-gray-700 border border-gray-600 rounded-lg shadow-xl z-10 max-h-60 overflow-y-auto">
                      {folders.map(folder => (
                        <button
                          key={folder}
                          onClick={() => {
                            selectFolder(folder)
                            setFolderDropdownOpen(false)
                          }}
                          disabled={isLoadingFiles}
                          className="w-full px-4 py-2 text-left text-sm hover:bg-gray-600 disabled:opacity-50 flex items-center gap-2 transition-colors"
                        >
                          <span className="text-yellow-400">📁</span> {folder}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                {isLoadingFiles && (
                  <span className="text-sm text-gray-400 flex items-center gap-2">
                    <span className="animate-spin">⏳</span> Loading files...
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Files Summary */}
          {isLoadingFiles ? (
            <div className="py-8 text-center text-gray-400">
              Loading files...
            </div>
          ) : files.length > 0 ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-gray-400">
                    <span className="text-white font-medium">{files.length}</span> 3D models found
                  </p>
                  <p className="text-sm text-gray-500">
                    Total size: {formatBytes(files.reduce((acc, f) => acc + f.size, 0))}
                  </p>
                </div>

                <div className="flex items-center gap-4">
                  {/* Batch size selector */}
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-gray-400">Batch:</label>
                    <select
                      value={batchSize}
                      onChange={(e) => setBatchSize(Number(e.target.value))}
                      className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-sm"
                      disabled={processing.isProcessing}
                    >
                      <option value={25}>25</option>
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                      <option value={200}>200</option>
                    </select>
                  </div>

                  {/* Clear index checkbox */}
                  <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={clearIndex}
                      onChange={(e) => setClearIndex(e.target.checked)}
                      disabled={processing.isProcessing}
                      className="rounded bg-gray-700 border-gray-600"
                    />
                    Clear existing
                  </label>

                  {/* Process button */}
                  <button
                    onClick={processFiles}
                    disabled={processing.isProcessing || files.length === 0}
                    className="px-6 py-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-medium transition-colors flex items-center gap-2"
                  >
                    {processing.isProcessing ? (
                      <>
                        <span className="animate-spin">⚙️</span>
                        Processing...
                      </>
                    ) : (
                      <>Process All</>
                    )}
                  </button>
                </div>
              </div>

              {/* File list preview */}
              <div className="max-h-48 overflow-y-auto bg-gray-700/50 rounded-lg border border-gray-600">
                <table className="w-full text-sm">
                  <thead className="bg-gray-700 sticky top-0">
                    <tr>
                      <th className="text-left px-4 py-2 text-gray-400 font-medium">Name</th>
                      <th className="text-left px-4 py-2 text-gray-400 font-medium">Format</th>
                      <th className="text-right px-4 py-2 text-gray-400 font-medium">Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {files.slice(0, 100).map(file => (
                      <tr key={file.key} className="border-t border-gray-600/50">
                        <td className="px-4 py-2 truncate max-w-xs">{file.name}</td>
                        <td className="px-4 py-2 text-gray-400">.{file.extension}</td>
                        <td className="px-4 py-2 text-right text-gray-400">{formatBytes(file.size)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {files.length > 100 && (
                  <div className="px-4 py-2 text-center text-gray-500 text-sm border-t border-gray-600/50">
                    ... and {files.length - 100} more files
                  </div>
                )}
              </div>
            </div>
          ) : currentPrefix ? (
            <div className="py-8 text-center text-gray-500">
              No 3D model files in this folder
            </div>
          ) : (
            <div className="py-8 text-center text-gray-500">
              Select a folder to browse models
            </div>
          )}
        </section>

        {/* Processing Progress */}
        {(processing.isProcessing || completionStats) && (
          <section className="bg-gray-800 rounded-xl border border-gray-700 p-6">
            <h2 className="text-lg font-semibold mb-4">
              {processing.isProcessing ? 'Processing' : 'Complete'}
            </h2>

            {processing.isProcessing ? (
              <div className="space-y-4">
                {/* Progress bar */}
                <div className="relative">
                  <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-600 to-blue-400 transition-all duration-300"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                  <div className="absolute inset-0 flex items-center justify-center text-xs font-medium">
                    {progressPercent.toFixed(1)}%
                  </div>
                </div>

                {/* Stats grid */}
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                  <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-blue-400">{processing.batch}/{processing.totalBatches}</div>
                    <div className="text-xs text-gray-400">Batches</div>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-green-400">{processing.processed}</div>
                    <div className="text-xs text-gray-400">Processed</div>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-red-400">{processing.failed}</div>
                    <div className="text-xs text-gray-400">Failed</div>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-yellow-400">{formatTime(elapsed)}</div>
                    <div className="text-xs text-gray-400">Elapsed</div>
                  </div>
                  <div className="bg-gray-700/50 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-purple-400">{formatTime(eta)}</div>
                    <div className="text-xs text-gray-400">ETA</div>
                  </div>
                </div>

                {/* Current batch info */}
                <p className="text-sm text-gray-400 text-center">
                  Processing batch {processing.batch} of {processing.totalBatches}
                  ({processing.processed + processing.failed}/{processing.total} models)
                </p>
              </div>
            ) : completionStats && (
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-green-400">{completionStats.added}</div>
                  <div className="text-sm text-gray-400">Models Added</div>
                </div>
                <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-red-400">{completionStats.failed}</div>
                  <div className="text-sm text-gray-400">Failed</div>
                </div>
                <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 text-center">
                  <div className="text-3xl font-bold text-blue-400">{formatTime(completionStats.time)}</div>
                  <div className="text-sm text-gray-400">Total Time</div>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Indexed Models */}
        {models.length > 0 && searchResults.length === 0 && (
          <section className="bg-gray-800 rounded-xl border border-gray-700 p-6">
            <h2 className="text-lg font-semibold mb-4">
              Indexed Models ({models.length})
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {models.slice(0, 24).map(model => (
                <div key={model.id} className="bg-gray-700/50 rounded-lg overflow-hidden border border-gray-600">
                  <div className="aspect-square">
                    <img
                      src={`${API_URL}/previews/${model.id}.jpg`}
                      alt={model.name}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none'
                      }}
                    />
                  </div>
                  <div className="p-2">
                    <h4 className="text-sm truncate" title={model.name}>{model.name}</h4>
                  </div>
                </div>
              ))}
            </div>
            {models.length > 24 && (
              <p className="text-center text-gray-500 text-sm mt-4">
                ... and {models.length - 24} more models
              </p>
            )}
          </section>
        )}
      </div>
    </div>
  )
}

export default App
