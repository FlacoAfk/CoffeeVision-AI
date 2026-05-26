import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: { Accept: 'application/json' },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const message =
        error.response.data?.error ||
        error.response.data?.message ||
        `Request failed with status ${error.response.status}`;
      return Promise.reject(new ApiError(message, error.response.status));
    }
    if (error.request) {
      return Promise.reject(
        new ApiError('No response from server. Check your connection.', 0),
      );
    }
    return Promise.reject(new ApiError(error.message, 0));
  },
);

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

export async function predictLeaf(file, { onUploadProgress } = {}) {
  const formData = new FormData();
  formData.append('image', file);
  const response = await api.post('/api/predict/leaf', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onUploadProgress && evt.total) {
        onUploadProgress(Math.round((evt.loaded * 100) / evt.total));
      }
    },
  });
  return response.data;
}

export async function predictGrain(file, { onUploadProgress } = {}) {
  const formData = new FormData();
  formData.append('image', file);
  const response = await api.post('/api/predict/grain', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (evt) => {
      if (onUploadProgress && evt.total) {
        onUploadProgress(Math.round((evt.loaded * 100) / evt.total));
      }
    },
  });
  return response.data;
}

export async function checkHealth() {
  const response = await api.get('/api/health');
  return response.data;
}

export async function getModelConfig() {
  const response = await api.get('/api/models/config');
  return response.data;
}

export async function updateModelConfig(config) {
  const response = await api.post('/api/models/config', config);
  return response.data;
}

export default api;
