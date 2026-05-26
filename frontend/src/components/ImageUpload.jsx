import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { predictLeaf, predictGrain } from '../services/api';
import './ImageUpload.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png'];
const MAX_FILE_SIZE = 10 * 1024 * 1024;

const MODE_CONFIG = {
  leaf: {
    title: 'Upload Coffee Leaf Image',
    description: 'Drag and drop a coffee leaf photo, or click to select. JPG and PNG only (max 10 MB).',
    buttonText: 'Analyze Leaf',
    loadingText: 'Analyzing leaf...',
  },
  grain: {
    title: 'Upload Coffee Grain Image',
    description: 'Drag and drop a coffee grain photo, or click to select. JPG and PNG only (max 10 MB).',
    buttonText: 'Analyze Grain',
    loadingText: 'Analyzing grain...',
  },
};

export default function ImageUpload({ mode, onResult, onError }) {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);

  const config = MODE_CONFIG[mode] || MODE_CONFIG.leaf;

  const onDrop = useCallback((acceptedFiles, rejections) => {
    setError(null);
    if (rejections.length > 0) {
      const err = rejections[0].errors[0];
      if (err?.code === 'file-too-large') setError('File too large. Maximum size is 10 MB.');
      else if (err?.code === 'file-invalid-type') setError('Invalid file type. Only JPG and PNG allowed.');
      else setError(err?.message || 'Invalid file.');
      return;
    }
    if (acceptedFiles.length === 0) return;
    const selected = acceptedFiles[0];
    if (selected.size > MAX_FILE_SIZE) { setError('File too large. Maximum size is 10 MB.'); return; }
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] },
    maxSize: MAX_FILE_SIZE,
    maxFiles: 1,
    multiple: false,
  });

  const handleAnalyze = async () => {
    if (!file) return;
    setUploading(true);
    setProgress(0);
    setError(null);
    try {
      const predictFn = mode === 'grain' ? predictGrain : predictLeaf;
      const result = await predictFn(file, {
        onUploadProgress: (p) => setProgress(p),
      });
      if (onResult) onResult(result, preview);
    } catch (err) {
      const msg = err.message || 'Analysis failed. Please try again.';
      setError(msg);
      if (onError) onError(msg);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  const handleClear = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setError(null);
    setProgress(0);
  };

  return (
    <div className="image-upload" role="region" aria-label="Image upload">
      <h3 className="image-upload__title">{config.title}</h3>
      <p className="image-upload__description">{config.description}</p>

      {!preview && (
        <div
          {...getRootProps()}
          className={`image-upload__dropzone ${isDragActive ? 'image-upload__dropzone--active' : ''}`}
          role="button"
          aria-label="Upload image drop zone"
          tabIndex={0}
        >
          <input {...getInputProps()} aria-label="Choose image file" />
          <div className="image-upload__icon" aria-hidden="true">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
          </div>
          <p className="image-upload__drop-text">
            {isDragActive ? 'Drop your image here' : 'Drag & drop or click to browse'}
          </p>
          <p className="image-upload__hint">JPG, PNG — Max 10 MB</p>
        </div>
      )}

      {preview && (
        <div className="image-upload__preview" aria-label="Image preview">
          <img src={preview} alt="Selected image preview" className="image-upload__preview-img" />
          <button
            className="image-upload__clear-btn"
            onClick={handleClear}
            aria-label="Remove selected image"
            disabled={uploading}
            type="button"
          >
            ✕
          </button>
        </div>
      )}

      {error && (
        <div className="image-upload__error" role="alert" aria-live="assertive">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <line x1="15" y1="9" x2="9" y2="15" />
            <line x1="9" y1="9" x2="15" y2="15" />
          </svg>
          <span>{error}</span>
        </div>
      )}

      {uploading && (
        <div className="image-upload__progress" role="progressbar" aria-valuenow={progress} aria-valuemin={0} aria-valuemax={100}>
          <div className="progress-bar">
            <div className="progress-bar-fill" style={{ width: `${progress}%`, backgroundColor: 'var(--color-primary-500)' }} />
          </div>
          <span className="image-upload__progress-text">{progress}%</span>
        </div>
      )}

      {preview && (
        <button
          className="btn btn-primary image-upload__submit"
          onClick={handleAnalyze}
          disabled={uploading || !file}
          type="button"
          aria-label="Analyze image"
        >
          {uploading ? (
            <><span className="spinner" aria-hidden="true" />{config.loadingText}</>
          ) : config.buttonText}
        </button>
      )}
    </div>
  );
}
