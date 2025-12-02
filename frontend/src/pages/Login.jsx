import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Lock, Mail, User, Eye, EyeOff, ShieldCheck } from 'lucide-react'
import { toast } from 'react-toastify'
import useAuthStore from '../utils/authStore'

function Login() {
  const navigate = useNavigate()
  const login = useAuthStore((state) => state.login)
  
  const [isLogin, setIsLogin] = useState(true)
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    role: 'analyst'
  })

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      // Demo authentication - no backend required
      if (isLogin) {
        // Demo login - accept any credentials
        if (!formData.email || !formData.password) {
          throw new Error('Please fill in all fields')
        }
        
        const demoUser = {
          id: 1,
          name: formData.name || 'Demo User',
          email: formData.email,
          role: 'analyst'
        }
        
        login(demoUser, 'demo-token')
        toast.success('Login successful!')
        navigate('/')
      } else {
        // Demo register - just create user and auto-login
        if (!formData.name || !formData.email || !formData.password) {
          throw new Error('Please fill in all fields')
        }
        
        const demoUser = {
          id: 1,
          name: formData.name,
          email: formData.email,
          role: formData.role
        }
        
        login(demoUser, 'demo-token')
        toast.success('Registration successful!')
        navigate('/')
      }
    } catch (error) {
      toast.error(error.message || 'Authentication failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center gradient-forensic p-4">
      <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10"></div>
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative w-full max-w-md"
      >
        {/* Logo and Title */}
        <div className="text-center mb-8">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
            className="inline-flex items-center justify-center w-20 h-20 bg-white rounded-full mb-4 shadow-lg"
          >
            <ShieldCheck className="w-10 h-10 text-primary-600" />
          </motion.div>
          <h1 className="text-4xl font-bold text-white mb-2">
            Bite Mark Analysis
          </h1>
          <p className="text-gray-300">Forensic Classification System</p>
        </div>

        {/* Login/Register Form */}
        <div className="card glass-effect shadow-2xl">
          <div className="flex mb-6 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setIsLogin(true)}
              className={`flex-1 py-2 rounded-md font-medium transition-all ${
                isLogin
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Login
            </button>
            <button
              onClick={() => setIsLogin(false)}
              className={`flex-1 py-2 rounded-md font-medium transition-all ${
                !isLogin
                  ? 'bg-white text-primary-600 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
              }`}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Full Name
                </label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className="input-field pl-10"
                    placeholder="John Doe"
                    required={!isLogin}
                  />
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Email Address
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className="input-field pl-10"
                  placeholder="you@example.com"
                  required
                />
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Password
              </label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type={showPassword ? 'text' : 'password'}
                  name="password"
                  value={formData.password}
                  onChange={handleChange}
                  className="input-field pl-10 pr-10"
                  placeholder="••••••••"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {!isLogin && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Role
                </label>
                <select
                  name="role"
                  value={formData.role}
                  onChange={handleChange}
                  className="input-field"
                >
                  <option value="analyst">Forensic Analyst</option>
                  <option value="investigator">Investigator</option>
                  <option value="admin">Administrator</option>
                </select>
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <div className="spinner w-5 h-5"></div>
                  Processing...
                </>
              ) : (
                <>{isLogin ? 'Sign In' : 'Create Account'}</>
              )}
            </button>

            {/* Quick Demo Login Button */}
            {isLogin && (
              <button
                type="button"
                onClick={() => {
                  setFormData({ ...formData, email: 'demo@forensics.com', password: 'demo123' });
                  setTimeout(() => {
                    document.querySelector('form').requestSubmit();
                  }, 100);
                }}
                disabled={loading}
                className="w-full mt-2 py-2 px-4 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-all font-medium"
              >
                Quick Demo Login
              </button>
            )}
          </form>

          {isLogin && (
            <div className="mt-4 text-center">
              <a href="#" className="text-sm text-primary-600 hover:text-primary-700">
                Forgot password?
              </a>
            </div>
          )}
        </div>

        {/* Demo Credentials */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-6 p-4 bg-white/10 backdrop-blur-md rounded-lg text-white text-sm"
        >
          <p className="font-medium mb-2">Demo Credentials:</p>
          <p>Email: demo@forensics.com</p>
          <p>Password: demo123</p>
        </motion.div>
      </motion.div>
    </div>
  )
}

export default Login
