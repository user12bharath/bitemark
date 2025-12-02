import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Database, Trash2, Server, RefreshCw, Activity, BarChart3 } from 'lucide-react'
import { toast } from 'react-toastify'
import { analysisAPI, modelAPI } from '../services/api'

function Settings() {
  const [backendStatus, setBackendStatus] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    checkBackendHealth()
    fetchModelInfo()
  }, [])

  const checkBackendHealth = async () => {
    try {
      const response = await analysisAPI.getHealthCheck()
      setBackendStatus(response.data)
    } catch (error) {
      setBackendStatus({ status: 'error', error: 'Backend unreachable' })
    }
  }

  const fetchModelInfo = async () => {
    try {
      const response = await modelAPI.getModelInfo()
      setModelInfo(response.data)
    } catch (error) {
      console.error('Failed to fetch model info:', error)
    }
  }

  const handleRefreshStatus = async () => {
    setLoading(true)
    await checkBackendHealth()
    await fetchModelInfo()
    setLoading(false)
    toast.success('Status refreshed!')
  }

  const handleClearData = () => {
    if (confirm('Are you sure you want to clear all analysis history? This action cannot be undone.')) {
      toast.success('Analysis history cleared (feature not implemented yet)')
    }
  }

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">System Settings</h1>
          <p className="text-gray-500 mt-1">Monitor backend status and model information</p>
        </div>
        <button
          onClick={handleRefreshStatus}
          disabled={loading}
          className="btn-primary flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh Status
        </button>
      </div>

      {/* Backend Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Server className="w-6 h-6 text-primary-600" />
          <h2 className="text-xl font-semibold text-gray-900">Backend Status</h2>
        </div>

        {backendStatus ? (
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-3">
                <div className={`w-3 h-3 rounded-full ${
                  backendStatus.status === 'healthy' ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="font-medium">API Status</span>
              </div>
              <span className="text-sm text-gray-600 capitalize">{backendStatus.status}</span>
            </div>
            
            {backendStatus.status === 'healthy' && (
              <>
                <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                  <span className="font-medium">Model Loaded</span>
                  <span className={`text-sm ${
                    backendStatus.model_loaded ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {backendStatus.model_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
                
                {backendStatus.preprocessor_loaded !== undefined && (
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Preprocessor Loaded</span>
                    <span className={`text-sm ${
                      backendStatus.preprocessor_loaded ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {backendStatus.preprocessor_loaded ? 'Yes' : 'No'}
                    </span>
                  </div>
                )}
                
                {backendStatus.uptime_seconds && (
                  <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    <span className="font-medium">Uptime</span>
                    <span className="text-sm text-gray-600">
                      {Math.round(backendStatus.uptime_seconds / 3600 * 100) / 100} hours
                    </span>
                  </div>
                )}
              </>
            )}
            
            {backendStatus.error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-sm text-red-600">Error: {backendStatus.error}</p>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center py-8">
            <Activity className="w-6 h-6 text-gray-400 animate-pulse" />
            <span className="ml-2 text-gray-500">Checking backend status...</span>
          </div>
        )}
      </motion.div>

      {/* Model Information */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <BarChart3 className="w-6 h-6 text-primary-600" />
          <h2 className="text-xl font-semibold text-gray-900">Model Information</h2>
        </div>

        {modelInfo ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modelInfo.api_version && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">API Version</div>
                <div className="font-medium">{modelInfo.api_version}</div>
              </div>
            )}
            
            {modelInfo.total_parameters && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Parameters</div>
                <div className="font-medium">{modelInfo.total_parameters.toLocaleString()}</div>
              </div>
            )}
            
            {modelInfo.input_shape && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Input Shape</div>
                <div className="font-medium">{JSON.stringify(modelInfo.input_shape)}</div>
              </div>
            )}
            
            {modelInfo.tensorflow_version && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">TensorFlow Version</div>
                <div className="font-medium">{modelInfo.tensorflow_version}</div>
              </div>
            )}
            
            {modelInfo.metrics?.test_accuracy && (
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-1">Test Accuracy</div>
                <div className="font-medium">{(modelInfo.metrics.test_accuracy * 100).toFixed(1)}%</div>
              </div>
            )}
            
            {modelInfo.class_names && (
              <div className="col-span-1 md:col-span-2 p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-500 mb-2">Supported Classes</div>
                <div className="flex flex-wrap gap-2">
                  {modelInfo.class_names.map((className, index) => (
                    <span key={index} className="px-2 py-1 bg-primary-100 text-primary-800 rounded-md text-sm">
                      {className}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="flex items-center justify-center py-8">
            <Activity className="w-6 h-6 text-gray-400 animate-pulse" />
            <span className="ml-2 text-gray-500">Loading model information...</span>
          </div>
        )}
      </motion.div>

      {/* Data Management */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="flex items-center gap-3 mb-6">
          <Database className="w-6 h-6 text-primary-600" />
          <h2 className="text-xl font-semibold text-gray-900">Data Management</h2>
        </div>

        <div className="space-y-4">
          <div className="p-4 border border-yellow-200 bg-yellow-50 rounded-lg">
            <h3 className="font-medium text-yellow-800 mb-2">Clear Analysis History</h3>
            <p className="text-sm text-yellow-700 mb-4">
              This will permanently delete all previous analysis results and cannot be undone.
            </p>
            <button
              onClick={handleClearData}
              className="btn-secondary text-red-600 border-red-200 hover:bg-red-50 flex items-center gap-2"
            >
              <Trash2 className="w-4 h-4" />
              Clear All Data
            </button>
          </div>

          <div className="p-4 border border-blue-200 bg-blue-50 rounded-lg">
            <h3 className="font-medium text-blue-800 mb-2">API Endpoints</h3>
            <p className="text-sm text-blue-700 mb-2">
              Backend API is running on: <code className="bg-blue-100 px-2 py-1 rounded">http://localhost:5000</code>
            </p>
            <div className="text-xs text-blue-600 space-y-1">
              <div>• POST /predict - Image classification</div>
              <div>• GET /health - Health check</div>
              <div>• GET /model/info - Model information</div>
              <div>• GET /analytics/stats - Usage statistics</div>
              <div>• GET /analytics/history - Analysis history</div>
              <div>• DELETE /analytics/history/&lt;id&gt; - Delete analysis</div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default Settings
