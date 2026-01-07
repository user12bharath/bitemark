import { useState, useEffect } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  ArrowLeft, 
  Image as ImageIcon, 
  Eye,
  Zap,
  BarChart3,
  Layers,
  Ruler,
  Target,
  User,
  RefreshCw,
  Download,
  Share2
} from 'lucide-react'
import { toast } from 'react-toastify'
import googleAnalysisAPI from '../services/googleAnalysisAPI'

function DetailedAnalysis() {
  const navigate = useNavigate()
  const location = useLocation()
  const [imageData, setImageData] = useState(null)
  const [detailedResult, setDetailedResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [user, setUser] = useState(null)

  useEffect(() => {
    // Get image data from navigation state
    if (location.state?.imageData) {
      setImageData(location.state.imageData)
    } else {
      // Redirect back if no image data
      navigate('/analysis')
    }

    initializeGoogleAPI()
  }, [location, navigate])

  const initializeGoogleAPI = async () => {
    try {
      await googleAnalysisAPI.init()
      const userInfo = googleAnalysisAPI.getUserInfo()
      setUser(userInfo)
    } catch (error) {
      console.error('Failed to initialize Google API:', error)
    }
  }

  const handleGoogleSignIn = async () => {
    try {
      const success = await googleAnalysisAPI.signIn()
      if (success) {
        const userInfo = googleAnalysisAPI.getUserInfo()
        setUser(userInfo)
        toast.success('Signed in successfully!')
      }
    } catch (error) {
      toast.error('Failed to sign in with Google')
    }
  }

  const handleDetailedAnalysis = async () => {
    if (!imageData) return

    try {
      setLoading(true)
      const result = await googleAnalysisAPI.getDetailedAnalysis(imageData)
      setDetailedResult(result)
      toast.success('Detailed analysis completed!')
    } catch (error) {
      console.error('Detailed analysis failed:', error)
      toast.error('Detailed analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleDownloadReport = () => {
    if (!detailedResult) return

    const reportData = {
      ...detailedResult,
      generatedBy: user?.email || 'anonymous',
      generatedAt: new Date().toISOString()
    }

    const blob = new Blob([JSON.stringify(reportData, null, 2)], {
      type: 'application/json'
    })

    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `detailed-analysis-${detailedResult.id}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    toast.success('Detailed report downloaded!')
  }

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getScoreBg = (score) => {
    if (score >= 80) return 'bg-green-500'
    if (score >= 60) return 'bg-yellow-500'
    return 'bg-red-500'
  }

  if (!imageData) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">No Image Data</h2>
          <button
            onClick={() => navigate('/analysis')}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Go to Analysis Page
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate(-1)}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <ArrowLeft size={20} />
              <span>Back</span>
            </button>
            <h1 className="text-3xl font-bold text-gray-800">Detailed Analysis</h1>
          </div>

          <div className="flex items-center space-x-4">
            {user ? (
              <div className="flex items-center space-x-2 text-sm text-gray-600">
                <User size={16} />
                <span>Signed in as {user.name}</span>
              </div>
            ) : (
              <button
                onClick={handleGoogleSignIn}
                className="flex items-center space-x-2 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
              >
                <User size={16} />
                <span>Sign in with Google</span>
              </button>
            )}

            {detailedResult && (
              <div className="flex space-x-2">
                <button
                  onClick={handleDownloadReport}
                  className="flex items-center space-x-2 bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors"
                >
                  <Download size={16} />
                  <span>Download Report</span>
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Image and Basic Info */}
          <div className="lg:col-span-1">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white rounded-xl shadow-sm border p-6"
            >
              <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                <ImageIcon className="mr-2" size={20} />
                Image Analysis
              </h3>

              {/* Basic Analysis Results */}
              {imageData && (
                <div className="space-y-4">
                  <div className="text-center">
                    <h4 className="text-xl font-bold text-gray-800">{imageData.prediction}</h4>
                    <p className="text-lg text-blue-600">{(imageData.confidence * 100).toFixed(1)}% confidence</p>
                  </div>

                  <div className="space-y-2">
                    <h5 className="font-medium text-gray-700">Class Probabilities:</h5>
                    {Object.entries(imageData.probabilities).map(([className, probability]) => (
                      <div key={className} className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="capitalize">{className}</span>
                          <span>{(probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-blue-500 transition-all duration-500"
                            style={{ width: `${probability * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <button
                    onClick={handleDetailedAnalysis}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center space-x-2"
                  >
                    {loading ? (
                      <>
                        <RefreshCw className="animate-spin" size={20} />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <Eye size={20} />
                        <span>Get Detailed Analysis</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </motion.div>
          </div>

          {/* Right Column - Detailed Analysis Results */}
          <div className="lg:col-span-2">
            {!detailedResult && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-xl shadow-sm border p-8 text-center"
              >
                <Eye className="mx-auto mb-4 text-gray-400" size={48} />
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Advanced Analysis</h3>
                <p className="text-gray-600 mb-6">
                  Click "Get Detailed Analysis" to perform advanced image analysis using Google's AI services.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-500">
                  <div className="flex flex-col items-center">
                    <Layers className="mb-2" size={24} />
                    <span>Edge Detection</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <BarChart3 className="mb-2" size={24} />
                    <span>Texture Analysis</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <Ruler className="mb-2" size={24} />
                    <span>Dimensional Analysis</span>
                  </div>
                  <div className="flex flex-col items-center">
                    <Target className="mb-2" size={24} />
                    <span>Comparative Analysis</span>
                  </div>
                </div>
              </motion.div>
            )}

            {loading && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-white rounded-xl shadow-sm border p-8 text-center"
              >
                <RefreshCw className="animate-spin mx-auto mb-4 text-blue-600" size={48} />
                <h3 className="text-xl font-semibold text-gray-800 mb-2">Processing Advanced Analysis</h3>
                <p className="text-gray-600">
                  Please wait while we perform detailed image analysis using Google's AI services...
                </p>
              </motion.div>
            )}

            {detailedResult && (
              <div className="space-y-6">
                {/* Edge Detection Analysis */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white rounded-xl shadow-sm border p-6"
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <Layers className="mr-2" size={20} />
                    Edge Detection Analysis
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-2xl font-bold mb-1 ${getScoreColor(detailedResult.advancedFeatures.edgeDetection.sharpness)}`}>
                        {detailedResult.advancedFeatures.edgeDetection.sharpness.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Sharpness</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold mb-1 text-blue-600">
                        {detailedResult.advancedFeatures.edgeDetection.contours}
                      </div>
                      <div className="text-sm text-gray-600">Contours</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-2xl font-bold mb-1 ${getScoreColor(detailedResult.advancedFeatures.edgeDetection.clarity)}`}>
                        {detailedResult.advancedFeatures.edgeDetection.clarity.toFixed(1)}%
                      </div>
                      <div className="text-sm text-gray-600">Clarity</div>
                    </div>
                  </div>
                </motion.div>

                {/* Texture Analysis */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="bg-white rounded-xl shadow-sm border p-6"
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <BarChart3 className="mr-2" size={20} />
                    Texture Analysis
                  </h3>
                  <div className="space-y-4">
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm text-gray-600">Roughness</span>
                        <span className="text-sm font-medium">{detailedResult.advancedFeatures.textureAnalysis.roughness.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${getScoreBg(detailedResult.advancedFeatures.textureAnalysis.roughness)}`}
                          style={{ width: `${detailedResult.advancedFeatures.textureAnalysis.roughness}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm text-gray-600">Uniformity</span>
                        <span className="text-sm font-medium">{detailedResult.advancedFeatures.textureAnalysis.uniformity.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${getScoreBg(detailedResult.advancedFeatures.textureAnalysis.uniformity)}`}
                          style={{ width: `${detailedResult.advancedFeatures.textureAnalysis.uniformity}%` }}
                        ></div>
                      </div>
                    </div>
                    <div className="bg-gray-50 p-3 rounded">
                      <span className="text-sm text-gray-600">Pattern Type: </span>
                      <span className="font-medium capitalize">{detailedResult.advancedFeatures.textureAnalysis.patterns}</span>
                    </div>
                  </div>
                </motion.div>

                {/* Dimensional Analysis */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="bg-white rounded-xl shadow-sm border p-6"
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <Ruler className="mr-2" size={20} />
                    Dimensional Analysis
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="bg-blue-50 p-4 rounded-lg text-center">
                      <div className="text-lg font-bold text-blue-600 mb-1">
                        {detailedResult.advancedFeatures.dimensionalAnalysis.estimatedDepth.toFixed(2)} mm
                      </div>
                      <div className="text-sm text-blue-700">Estimated Depth</div>
                    </div>
                    <div className="bg-green-50 p-4 rounded-lg text-center">
                      <div className="text-lg font-bold text-green-600 mb-1">
                        {detailedResult.advancedFeatures.dimensionalAnalysis.surfaceArea.toFixed(1)} mm²
                      </div>
                      <div className="text-sm text-green-700">Surface Area</div>
                    </div>
                    <div className="bg-purple-50 p-4 rounded-lg text-center">
                      <div className="text-lg font-bold text-purple-600 mb-1">
                        {detailedResult.advancedFeatures.dimensionalAnalysis.volume.toFixed(2)} mm³
                      </div>
                      <div className="text-sm text-purple-700">Volume</div>
                    </div>
                  </div>
                </motion.div>

                {/* Comparative Analysis */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="bg-white rounded-xl shadow-sm border p-6"
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <Target className="mr-2" size={20} />
                    Comparative Analysis
                  </h3>
                  <div className="space-y-4">
                    <div className="space-y-3">
                      {Object.entries({
                        Human: detailedResult.advancedFeatures.comparativeAnalysis.similarityToHuman,
                        Dog: detailedResult.advancedFeatures.comparativeAnalysis.similarityToDog,
                        Snake: detailedResult.advancedFeatures.comparativeAnalysis.similarityToSnake
                      }).map(([type, similarity]) => (
                        <div key={type}>
                          <div className="flex justify-between mb-1">
                            <span className="text-sm font-medium text-gray-700">Similarity to {type}</span>
                            <span className="text-sm text-gray-600">{similarity.toFixed(1)}%</span>
                          </div>
                          <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                              className={`h-2 rounded-full ${getScoreBg(similarity)}`}
                              style={{ width: `${similarity}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                    
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h4 className="font-medium text-gray-800 mb-2">Unique Features Detected:</h4>
                      <ul className="space-y-1">
                        {detailedResult.advancedFeatures.comparativeAnalysis.uniqueFeatures.map((feature, index) => (
                          <li key={index} className="text-sm text-gray-600 flex items-center">
                            <span className="w-1.5 h-1.5 bg-blue-500 rounded-full mr-2"></span>
                            {feature}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </motion.div>

                {/* Analysis Metadata */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="bg-white rounded-xl shadow-sm border p-6"
                >
                  <h3 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                    <Zap className="mr-2" size={20} />
                    Analysis Metadata
                  </h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600">Analysis Time:</span>
                      <div className="font-medium">{new Date(detailedResult.metadata.analysisTime).toLocaleString()}</div>
                    </div>
                    <div>
                      <span className="text-gray-600">API Version:</span>
                      <div className="font-medium">{detailedResult.metadata.apiVersion}</div>
                    </div>
                    <div>
                      <span className="text-gray-600">Overall Confidence:</span>
                      <div className={`font-medium ${getScoreColor(detailedResult.metadata.confidence * 100)}`}>
                        {(detailedResult.metadata.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div>
                      <span className="text-gray-600">Analysis ID:</span>
                      <div className="font-medium text-xs">{detailedResult.id}</div>
                    </div>
                  </div>
                </motion.div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default DetailedAnalysis