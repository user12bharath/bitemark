# Bite Mark Classification - Frontend

## 🎨 Overview

Modern, responsive frontend application for the Bite Mark Classification System. Built with React, Vite, and Tailwind CSS for a seamless forensic analysis experience.

## ✨ Features

- **🔐 Authentication**: Secure login/register system
- **📊 Dashboard**: Real-time statistics and analytics
- **🔍 Image Analysis**: Drag-and-drop image upload with instant classification
- **📜 History**: Complete analysis history with search and filtering
- **📈 Model Metrics**: Detailed performance metrics and visualizations
- **⚙️ Settings**: User profile and preference management
- **🎨 Responsive Design**: Works on desktop, tablet, and mobile
- **🌓 Modern UI**: Clean, professional interface with smooth animations

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ and npm/yarn
- Backend API running on `http://localhost:5000`

### Installation

```bash
# Install dependencies
npm install

# Copy environment configuration
cp .env.example .env

# Start development server
npm run dev
```

The application will be available at `http://localhost:3000`

## 📁 Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable components
│   │   └── Layout.jsx   # Main layout with sidebar
│   ├── pages/          # Page components
│   │   ├── Login.jsx
│   │   ├── Dashboard.jsx
│   │   ├── Analysis.jsx
│   │   ├── History.jsx
│   │   ├── ModelMetrics.jsx
│   │   └── Settings.jsx
│   ├── services/       # API services
│   │   └── api.js      # Axios configuration
│   ├── utils/          # Utility functions
│   │   └── authStore.js # Authentication state
│   ├── styles/         # Global styles
│   │   └── index.css   # Tailwind imports
│   ├── App.jsx         # Main app component
│   └── main.jsx        # Entry point
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## 🛠️ Available Scripts

```bash
# Development
npm run dev          # Start dev server with hot reload

# Production
npm run build        # Build for production
npm run preview      # Preview production build

# Linting
npm run lint         # Run ESLint
```

## 🎨 Tech Stack

- **React 18** - UI framework
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **Zustand** - State management
- **Axios** - HTTP client
- **Recharts** - Data visualization
- **Framer Motion** - Animations
- **React Dropzone** - File upload
- **React Toastify** - Notifications
- **Lucide React** - Icon library

## 🔌 API Integration

The frontend communicates with the backend API for:

- **Authentication**: Login, register, logout
- **Analysis**: Upload images, get predictions
- **History**: Fetch, search, delete analyses
- **Metrics**: Model performance data
- **Stats**: Dashboard statistics

API configuration in `src/services/api.js`

## 🎯 Key Features Explained

### Authentication
- JWT-based authentication
- Persistent login with local storage
- Protected routes
- Demo credentials available

### Image Analysis
- Drag-and-drop upload
- Real-time prediction
- Confidence scores
- Probability distribution visualization

### Dashboard
- Overview statistics
- Weekly trend charts
- Class distribution pie chart
- Recent analyses

### Model Metrics
- Overall accuracy, precision, recall, F1
- Per-class performance
- Training history graphs
- Confusion matrix visualization

### History
- Searchable analysis records
- Filter by class
- View, download, delete operations
- Detailed analysis view

## 🎨 Customization

### Colors

Edit `tailwind.config.js` to customize the color scheme:

```js
colors: {
  primary: { /* your colors */ },
  forensic: { /* theme colors */ }
}
```

### API Endpoint

Update `.env` file:

```
VITE_API_URL=http://your-api-url/api
```

## 📱 Responsive Design

- Mobile-first approach
- Breakpoints: sm (640px), md (768px), lg (1024px), xl (1280px)
- Flexible layouts and components
- Touch-friendly interactions

## 🔒 Security

- Token-based authentication
- Secure HTTP-only cookies (backend)
- CSRF protection
- Input validation
- XSS prevention

## 🚧 Production Build

```bash
# Build optimized production bundle
npm run build

# Preview production build locally
npm run preview

# Deploy the 'dist' folder to your hosting service
```

## 📝 Environment Variables

Create a `.env` file:

```env
VITE_API_URL=http://localhost:5000/api
```

## 🤝 Integration with Backend

This frontend is designed to work with the Python Flask/FastAPI backend. Ensure:

1. Backend is running on the configured port
2. CORS is enabled for the frontend origin
3. API endpoints match the expected structure

## 📄 License

Part of the Bite Mark Classification System - Forensic Research License

## 🆘 Troubleshooting

### API Connection Issues
- Check if backend is running
- Verify `VITE_API_URL` in `.env`
- Check browser console for errors

### Build Errors
- Clear node_modules: `rm -rf node_modules && npm install`
- Clear cache: `rm -rf dist .vite`
- Update dependencies: `npm update`

### Styling Issues
- Rebuild Tailwind: `npm run build`
- Check PostCSS config
- Verify Tailwind imports in `index.css`

## 📞 Support

For issues and questions, please refer to the main project documentation.
