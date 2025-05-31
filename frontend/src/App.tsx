import './App.css'
import { Chat } from './pages/chat/chat'
import AdminPanel from './pages/admin' 
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './context/ThemeContext'

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="w-full h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
          <Routes>
            <Route path="/" element={<Chat />} />
            {/* <Route path="/admin" element={<AdminPanel />} /> */}
          </Routes>
        </div>
      </Router>
    </ThemeProvider>
  )
}

export default App;