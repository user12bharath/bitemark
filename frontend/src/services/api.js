import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor (simplified - no auth for now)
api.interceptors.request.use(
  (config) => {
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// No auth APIs needed for now

// Analysis APIs
export const analysisAPI = {
  uploadImage: (formData) => api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }),
  getAnalysisHistory: (params) => api.get('/analytics/history', { params }),
  getHealthCheck: () => api.get('/health'),
  deleteAnalysis: (id) => api.delete(`/analytics/history/${id}`),
}

// Model APIs
export const modelAPI = {
  getModelInfo: () => api.get('/model/info'),
  getMetrics: () => api.get('/model/info'), // Same endpoint provides metrics
}

// Stats APIs
export const statsAPI = {
  getDashboardStats: () => api.get('/analytics/stats'),
  getRecentAnalyses: (limit = 10) => api.get(`/analytics/history?limit=${limit}`),
}

export default api
