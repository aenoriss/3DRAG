// TypeScript declarations for File System Access API (drag & drop)

interface FileSystemEntry {
  isFile: boolean
  isDirectory: boolean
  name: string
}

interface FileSystemFileEntry extends FileSystemEntry {
  file(successCallback: (file: File) => void): void
}

interface FileSystemDirectoryEntry extends FileSystemEntry {
  createReader(): FileSystemDirectoryReader
}

interface FileSystemDirectoryReader {
  readEntries(
    successCallback: (entries: FileSystemEntry[]) => void,
    errorCallback?: (error: DOMException) => void
  ): void
}

interface DataTransferItem {
  webkitGetAsEntry(): FileSystemEntry | null
}
