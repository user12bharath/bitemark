import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  TrendingUp, 
  Image as ImageIcon, 
  Activity, 
  Award,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react'
import { LineChart, Line, PieChart, Pie, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'
import { statsAPI } from '../services/api'
import { toast } from 'react-toastify'

const COLORS = ['#0ea5e9', '#10b981', '#ef4444']  // Blue, Green, Red for Human, Dog, Snake

function Dashboard() {
  const [stats, setStats] = useState(null)
  const [recentAnalyses, setRecentAnalyses] = useState([])
  const [classDistribution, setClassDistribution] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchDashboardData()
  }, [])

  const fetchDashboardData = async () => {
    try {
      const [statsRes, recentRes] = await Promise.all([
        statsAPI.getDashboardStats(),
        statsAPI.getRecentAnalyses(5)
      ])
      
      // Transform backend analytics response to frontend format
      const backendStats = statsRes.data
      const transformedStats = {
        totalAnalyses: backendStats.api_stats?.total_predictions || 0,
        todayAnalyses: backendStats.api_stats?.successful_predictions || 0,
        accuracy: parseFloat(((backendStats.model_performance?.test_accuracy || 0) * 100).toFixed(2)),
        avgProcessingTime: backendStats.predictions_per_hour || 0,
        successRate: backendStats.success_rate || 0,
        uptime: backendStats.uptime_hours || 0
      }
      
      setStats(transformedStats)
      
      // Format recent analyses timestamps - backend returns {history: [...], total_count: ..., returned_count: ...}
      const recentHistory = recentRes.data.history || []
      const formattedRecent = recentHistory.map(item => ({
        ...item,
        image: item.filename || `analysis_${item.id}.jpg`,
        timestamp: item.timestamp ? new Date(item.timestamp).toLocaleString('en-US', {
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit',
          hour12: false
        }).replace(',', '') : 'N/A',
        // Extract prediction details from the prediction object
        prediction: item.prediction?.predicted_class || 'Unknown',
        confidence: item.prediction?.confidence || 0
      }))
      
      // Create mock class distribution data since backend doesn't provide it yet
      const mockClassDistribution = [
        { name: 'Human', value: 45, color: COLORS[0] },
        { name: 'Dog', value: 35, color: COLORS[1] },
        { name: 'Snake', value: 20, color: COLORS[2] }
      ]
      
      setRecentAnalyses(formattedRecent)
      setClassDistribution(mockClassDistribution)
      
      if (formattedRecent.length === 0) {
        toast.info('No recent analyses. Start by uploading and analyzing images.')
      }
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
      toast.error('Failed to load dashboard. Backend may not be running.')
      
      // Set empty/minimal data instead of mock data
      setStats({
        totalAnalyses: 0,
        todayAnalyses: 0,
        accuracy: 0,
        avgProcessingTime: 0
      })
      setRecentAnalyses([])
      setClassDistribution([])
    } finally {
      setLoading(false)
    }
  }

  const statCards = [
    {
      title: 'Total Analyses',
      value: stats?.totalAnalyses || 0,
      icon: ImageIcon,
      color: 'blue',
      change: '+12.5%'
    },
    {
      title: 'Today\'s Analyses',
      value: stats?.todayAnalyses || 0,
      icon: TrendingUp,
      color: 'green',
      change: '+5'
    },
    {
      title: 'Model Accuracy',
      value: `${(stats?.accuracy || 0).toFixed(2)}%`,
      icon: Award,
      color: 'purple',
      change: '+2.1%'
    },
    {
      title: 'Avg Processing',
      value: `${stats?.avgProcessingTime || 0}s`,
      icon: Clock,
      color: 'orange',
      change: '-0.3s'
    },
  ]

  const recentTrends = [
    { date: 'Mon', analyses: 42 },
    { date: 'Tue', analyses: 38 },
    { date: 'Wed', analyses: 51 },
    { date: 'Thu', analyses: 45 },
    { date: 'Fri', analyses: 58 },
    { date: 'Sat', analyses: 35 },
    { date: 'Sun', analyses: 28 },
  ]

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
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">Overview of your forensic analysis system</p>
      </div>

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {statCards.map((stat, index) => (
          <motion.div
            key={stat.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card-hover"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600">{stat.title}</p>
                <p className="text-3xl font-bold text-gray-900 mt-2">{stat.value}</p>
                <p className="text-sm text-green-600 mt-2">
                  {stat.change} from last week
                </p>
              </div>
              <div className={`p-3 bg-${stat.color}-100 rounded-lg`}>
                <stat.icon className={`w-6 h-6 text-${stat.color}-600`} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Weekly Trend */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Weekly Analysis Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={recentTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="analyses" 
                stroke="#0ea5e9" 
                strokeWidth={2}
                dot={{ fill: '#0ea5e9', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Class Distribution */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="card"
        >
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Class Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={classDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {classDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Recent Analyses */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card"
      >
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-gray-900">Recent Analyses</h3>
          <a href="/history" className="text-primary-600 hover:text-primary-700 text-sm font-medium">
            View All →
          </a>
        </div>

        <div className="space-y-4">
          {recentAnalyses.map((analysis, index) => (
            <motion.div
              key={analysis.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center justify-between p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            >
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-gray-300 rounded-lg flex items-center justify-center">
                  <ImageIcon className="w-6 h-6 text-gray-600" />
                </div>
                <div>
                  <p className="font-medium text-gray-900">{analysis.image}</p>
                  <p className="text-sm text-gray-500">{analysis.timestamp}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className="font-medium text-gray-900">{analysis.prediction}</p>
                  <p className="text-sm text-gray-500">
                    {(analysis.confidence * 100).toFixed(2)}% confidence
                  </p>
                </div>
                {analysis.confidence > 0.9 ? (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                ) : (
                  <AlertCircle className="w-6 h-6 text-yellow-500" />
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <motion.a
          href="/analysis"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card-hover text-center gradient-primary text-white"
        >
          <Activity className="w-12 h-12 mx-auto mb-3" />
          <h4 className="text-lg font-semibold mb-2">New Analysis</h4>
          <p className="text-sm opacity-90">Upload and analyze a new bite mark image</p>
        </motion.a>

        <motion.a
          href="/metrics"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card-hover text-center"
        >
          <Award className="w-12 h-12 mx-auto mb-3 text-purple-600" />
          <h4 className="text-lg font-semibold mb-2">Model Metrics</h4>
          <p className="text-sm text-gray-600">View detailed performance metrics</p>
        </motion.a>

        <motion.a
          href="/history"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card-hover text-center"
        >
          <Clock className="w-12 h-12 mx-auto mb-3 text-orange-600" />
          <h4 className="text-lg font-semibold mb-2">Analysis History</h4>
          <p className="text-sm text-gray-600">Browse all past analyses</p>
        </motion.a>
      </div>
    </div>
  )
}

export default Dashboard
