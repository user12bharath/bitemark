import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const useAuthStore = create(
  persist(
    (set) => ({
      isAuthenticated: false, // Start as false to require login
      user: null,
      token: null,
      
      login: (userData, token = null) => {
        set({ 
          isAuthenticated: true, 
          user: userData,
          token: token 
        })
      },
      
      logout: () => {
        set({ 
          isAuthenticated: false, 
          user: null,
          token: null 
        })
      },
      
      updateUser: (userData) => {
        set({ user: userData })
      },
    }),
    {
      name: 'auth-storage',
    }
  )
)

export default useAuthStore
