import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ToastContainer } from 'react-toastify'
import 'react-toastify/dist/ReactToastify.css'

import Dashboard from './pages/Dashboard'
import Analysis from './pages/Analysis'
import History from './pages/History'
import ModelMetrics from './pages/ModelMetrics'
import Settings from './pages/Settings'
import Login from './pages/Login'
import Layout from './components/Layout'
import useAuthStore from './utils/authStore'

function App() {
  const { isAuthenticated } = useAuthStore()

  return (
    <Router>
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={true}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="light"
      />
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route
          path="/"
          element={
            isAuthenticated ? <Layout /> : <Navigate to="/login" />
          }
        >
          <Route index element={<Dashboard />} />
          <Route path="analysis" element={<Analysis />} />
          <Route path="history" element={<History />} />
          <Route path="metrics" element={<ModelMetrics />} />
          <Route path="settings" element={<Settings />} />
        </Route>
        <Route path="*" element={<Navigate to={isAuthenticated ? "/" : "/login"} />} />
      </Routes>
    </Router>
  )
}

export default App
