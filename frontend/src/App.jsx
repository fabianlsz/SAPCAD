import React, { useState } from 'react';
import { Toaster } from 'react-hot-toast';
import { GlobalProvider } from './states/GlobalState';
import ChatOllamaConnector from './components/ChatOllamaConnector';
import InteractiveParticleSystem from './components/general/InteractiveParticleSystem.jsx';
import Result from './components/result/index.jsx';
import './App.css';

function App() {
    const [result, setResult] = useState(null);
    const [showComparison, setShowComparison] = useState(false);

    return (
        <GlobalProvider>
            <InteractiveParticleSystem />
            <Toaster
                position="top-right"
                toastOptions={{
                    style: {
                        background: '#1f2937',
                        color: 'white',
                        border: '1px solid rgba(255,255,255,0.1)',
                        boxShadow: '0 0 20px rgba(34, 211, 238, 0.5)',
                    },
                    success: {
                        style: {
                            background: '#059669',
                            boxShadow: '0 0 15px #34d399',
                        },
                    },
                    error: {
                        style: {
                            background: '#7f1d1d',
                            boxShadow: '0 0 15px #f87171',
                        },
                    },
                }}
            />

            <div className="flex h-screen p-6 relative">
                {/* Left Panel - Chat interface */}
                <div className="h-full w-1/3 pr-4 bg-transparent relative">
                    <ChatOllamaConnector
                        result={result}
                        setResult={setResult}
                        showComparison={showComparison}
                        setShowComparison={setShowComparison}
                    />
                </div>

                {/* Right Panel - Result component with viewer and upload */}
                <div className="h-full w-2/3 relative overflow-auto">
                    <Result />
                </div>
            </div>
        </GlobalProvider>
    );
}

export default App;
