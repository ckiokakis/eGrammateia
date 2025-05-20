import { useState, useEffect, useCallback } from "react";
import { Trash2, RefreshCcw } from "lucide-react";

export default function AdminPanel() {
  const [files, setFiles] = useState<string[]>([]);

  const fetchFiles = useCallback(async () => {
    try {
      const res = await fetch('/api/files/list');
      const data = await res.json();
      setFiles(data.files);
    } catch (err) {
      console.error('Error fetching files:', err);
    }
  }, []);

  useEffect(() => {
    fetchFiles();
  }, [fetchFiles]);

  // Handle drag over to allow drop
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  // Handle dropped files
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const droppedFiles = Array.from(e.dataTransfer.files);
    const formData = new FormData();
    droppedFiles.forEach((file) => formData.append('files', file));

    try {
      await fetch('/api/files/upload', {
        method: 'POST',
        body: formData,
      });
      fetchFiles();
    } catch (err) {
      console.error('Upload error:', err);
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await fetch('/api/files/delete', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename }),
      });
      fetchFiles();
    } catch (err) {
      console.error('Delete error:', err);
    }
  };

  return (
    <div className="flex flex-col min-w-0 h-dvh bg-background p-6 md:max-w-3xl mx-auto">
      <h1 className="text-2xl font-semibold mb-4">Admin Panel</h1>

      {/* Drag & Drop Box */}
      <div
        className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center mb-6 hover:border-gray-500 transition-colors"
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <p className="text-gray-500">Drag & drop files here to upload</p>
      </div>

      {/* Files Table */}
      <table className="w-full text-left border-collapse mb-4">
        <thead>
          <tr>
            <th className="pb-2">Filename</th>
            <th className="pb-2 w-12">Actions</th>
          </tr>
        </thead>
        <tbody>
          {files.map((file) => (
            <tr key={file} className="border-t">
              <td className="py-2">{file}</td>
              <td className="py-2">
                <button onClick={() => handleDelete(file)} aria-label="Delete file">
                  <Trash2 className="h-5 w-5 text-gray-600 hover:text-red-600" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* Refresh Button */}
      <button
        className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg hover:border-gray-500 transition-colors"
        onClick={() => { /* no-op for now */ }}
      >
        <RefreshCcw className="h-5 w-5" />
        Refresh
      </button>
    </div>
  );
}
