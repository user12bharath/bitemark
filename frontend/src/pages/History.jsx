import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Search, Filter, Download, Trash2, Eye, Calendar, Image as ImageIcon } from 'lucide-react'
import { toast } from 'react-toastify'
import { analysisAPI } from '../services/api'

function History() {
  const [analyses, setAnalyses] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchTerm, setSearchTerm] = useState('')
  const [filterClass, setFilterClass] = useState('all')
  const [selectedAnalysis, setSelectedAnalysis] = useState(null)

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      const response = await analysisAPI.getAnalysisHistory()
      // Backend returns {history: [...], total_count: ..., returned_count: ...}
      const historyData = response.data.history || []
      const formattedData = historyData.map(item => ({
        ...item,
        timestamp: item.timestamp ? new Date(item.timestamp).toLocaleString('en-US', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: false
        }).replace(',', '') : 'N/A',
        // Extract prediction details from the prediction object
        prediction: item.prediction?.predicted_class || 'Unknown',
        confidence: item.prediction?.confidence || 0,
        originalPrediction: item.prediction // Keep the full object for details view
      }))
      setAnalyses(formattedData)
      
      if (formattedData.length === 0) {
        toast.info('No analysis history yet. Upload and analyze images to see them here.')
      }
    } catch (error) {
      console.error('Failed to load history:', error)
      toast.error('Failed to load history. Backend may not be running.')
      setAnalyses([])
    } finally {
      setLoading(false)
    }
  }

  const handleDelete = async (id) => {
    if (!confirm('Are you sure you want to delete this analysis?')) return
    
    try {
      await analysisAPI.deleteAnalysis(id)
      setAnalyses(analyses.filter(a => a.id !== id))
      toast.success('Analysis deleted successfully')
    } catch (error) {
      toast.error('Failed to delete analysis')
    }
  }

  const filteredAnalyses = analyses.filter(analysis => {
    const matchesSearch = analysis.filename.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesFilter = filterClass === 'all' || analysis.prediction?.toLowerCase() === filterClass.toLowerCase()
    return matchesSearch && matchesFilter
  })

  const getConfidenceBadge = (confidence) => {
    if (confidence >= 0.9) return 'badge-success'
    if (confidence >= 0.7) return 'badge-warning'
    return 'badge-error'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-12 h-12"></div>
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analysis History</h1>
          <p className="text-gray-500 mt-1">{analyses.length} total analyses</p>
        </div>
        <button className="btn-primary flex items-center gap-2">
          <Download className="w-4 h-4" />
          Export All
        </button>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="input-field pl-10"
            />
          </div>

          {/* Filter */}
          <div className="relative">
            <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
            <select
              value={filterClass}
              onChange={(e) => setFilterClass(e.target.value)}
              className="input-field pl-10"
            >
              <option value="all">All Classes</option>
              <option value="human">Human</option>
              <option value="dog">Dog</option>
              <option value="snake">Snake</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results */}
      {filteredAnalyses.length === 0 ? (
        <div className="card text-center py-12">
          <ImageIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No analyses found</h3>
          <p className="text-gray-500">Try adjusting your search or filters</p>
        </div>
      ) : (
        <div className="card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Filename
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Prediction
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Timestamp
                  </th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredAnalyses.map((analysis, index) => (
                  <motion.tr
                    key={analysis.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="hover:bg-gray-50"
                  >
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gray-200 rounded flex items-center justify-center">
                          <ImageIcon className="w-5 h-5 text-gray-500" />
                        </div>
                        <span className="text-sm font-medium text-gray-900">
                          {analysis.filename}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className="text-sm font-medium text-gray-900">
                        {analysis.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`badge ${getConfidenceBadge(analysis.confidence)}`}>
                        {(analysis.confidence * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <Calendar className="w-4 h-4" />
                        {analysis.timestamp}
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => setSelectedAnalysis(analysis)}
                          className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                          title="View Details"
                        >
                          <Eye className="w-4 h-4" />
                        </button>
                        <button
                          className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleDelete(analysis.id)}
                          className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Detail Modal */}
      {selectedAnalysis && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedAnalysis(null)}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            onClick={(e) => e.stopPropagation()}
            className="card max-w-2xl w-full"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">Analysis Details</h3>
              <button
                onClick={() => setSelectedAnalysis(null)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <Eye className="w-5 h-5 text-gray-500" />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <p className="text-sm text-gray-600">Filename</p>
                <p className="text-lg font-medium text-gray-900">{selectedAnalysis.filename}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Prediction</p>
                <p className="text-lg font-medium text-gray-900">{selectedAnalysis.prediction}</p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Confidence</p>
                <p className="text-lg font-medium text-gray-900">
                  {(selectedAnalysis.confidence * 100).toFixed(2)}%
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Timestamp</p>
                <p className="text-lg font-medium text-gray-900">{selectedAnalysis.timestamp}</p>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </div>
  )
}

export default History
