import React, { useState, useCallback } from 'react';
import { Upload, ShieldCheck, Cpu, Zap, Activity, Info, AlertCircle, CheckCircle2, Shield, Settings2 } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import './index.css';

interface OCRData {
  data: any;
  image: string;
}

const App: React.FC = () => {
  const [isScanning, setIsScanning] = useState(false);
  const [ocrResult, setOcrResult] = useState<OCRData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lowGraphics, setLowGraphics] = useState(false);

  const onFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsScanning(true);
    setOcrResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:8000/ocr', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setOcrResult(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to process image');
    } finally {
      setIsScanning(false);
    }
  }, []);

  return (
    <div className={`app-container ${lowGraphics ? 'low-graphics' : ''}`}>
      <div className="bg-grid" />
      
      <header className="glass">
        <div className="logo">EGYPT ID <span style={{opacity: 0.5}}>— OCR.SYSTEM.v1</span></div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <Activity size={14} /> PIPELINE STABLE
          </div>
          <button 
            onClick={() => setLowGraphics(!lowGraphics)}
            style={{ padding: '0.5rem', background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-secondary)' }}
          >
            <Settings2 size={18} />
          </button>
        </div>
      </header>

      <main className="main-content">
        {/* Left Panel: Scanner */}
        <div className={`panel glass ${isScanning ? 'scanning' : ''}`}>
          <div className="scan-line" />
          
          <AnimatePresence mode="wait">
            {!ocrResult && !isScanning ? (
              <motion.div 
                key="upload"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                style={{ height: '100%' }}
              >
                <label className="upload-zone">
                  <input type="file" hidden onChange={onFileUpload} accept="image/*" />
                  <motion.div
                    animate={{ y: [0, -10, 0] }}
                    transition={{ repeat: Infinity, duration: 4 }}
                  >
                    <Upload size={48} color="var(--accent-primary)" />
                  </motion.div>
                  <h3 style={{ marginTop: '1.5rem', color: 'white' }}>Initialize Scan</h3>
                  <p style={{ color: 'var(--text-secondary)', marginTop: '0.5rem', textAlign: 'center' }}>
                    Drop Egyptian National ID image or click to browse
                  </p>
                </label>
              </motion.div>
            ) : (
              <motion.div 
                key="preview"
                initial={{ scale: 0.95, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                style={{ height: '100%', display: 'flex', flexDirection: 'column' }}
              >
                <div style={{ flex: 1, position: 'relative', overflow: 'hidden', borderRadius: '1rem' }}>
                  {ocrResult && <img src={ocrResult.image} className="id-preview" alt="ID Preview" />}
                  {isScanning && (
                    <div style={{ 
                      position: 'absolute', top: 0, left: 0, width: 100, height: 100,
                      background: 'rgba(0, 242, 255, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center'
                    }}>
                       <Cpu className="spin" />
                    </div>
                  )}
                </div>
                {ocrResult && (
                  <button onClick={() => setOcrResult(null)} style={{ marginTop: '1rem', background: 'transparent', border: '1px solid var(--border-color)' }}>
                    Scan Another
                  </button>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right Panel: Data */}
        <div className="panel glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1.5rem' }}>
            <h2 style={{ fontSize: '1.2rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldCheck size={20} color="var(--accent-primary)" />
              Extraction Results
            </h2>
            {ocrResult && (
              <div style={{ color: 'var(--success)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                <CheckCircle2 size={16} /> CONFIDENCE: {(ocrResult.data.confidence * 100).toFixed(0)}%
              </div>
            )}
          </div>

          <div className="result-data">
            {isScanning ? (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', gap: '1rem' }}>
                <motion.div
                  animate={{ rotate: 360 }}
                  transition={{ repeat: Infinity, duration: 2, ease: "linear" }}
                >
                  <Cpu size={32} color="var(--accent-primary)" />
                </motion.div>
                <p style={{ color: 'var(--accent-primary)', letterSpacing: '0.1em' }}>PROCESSING NEURAL PIPELINE...</p>
              </div>
            ) : error ? (
              <div style={{ color: 'var(--error)', display: 'flex', gap: '0.5rem' }}>
                <AlertCircle size={20} /> {error}
              </div>
            ) : ocrResult ? (
              <pre>
                {renderJSON(ocrResult.data)}
              </pre>
            ) : (
              <div style={{ opacity: 0.3, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <Shield size={64} style={{ marginBottom: '1rem' }} />
                <p>Waiting for Secure Input...</p>
              </div>
            )}
          </div>

          {ocrResult && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', borderRadius: '0.75rem', background: 'rgba(0,242,255,0.05)', border: '1px solid rgba(0,242,255,0.1)' }}>
               <div style={{ fontSize: '0.7rem', color: 'var(--accent-primary)', marginBottom: '0.5rem' }}>SYSTEM DIAGNOSTICS</div>
               <div style={{ display: 'flex', gap: '2rem', fontSize: '0.8rem' }}>
                  <div>Latency: <span style={{ color: 'white' }}>{ocrResult.data.processing_time_ms}ms</span></div>
                  <div>Engine: <span style={{ color: 'white' }}>EasyOCR</span></div>
                  <div>Mode: <span style={{ color: 'white' }}>Classical Fallback</span></div>
               </div>
            </div>
          )}
        </div>
      </main>

      <footer style={{ marginTop: '2rem', textAlign: 'center', fontSize: '0.7rem', color: 'var(--text-secondary)', letterSpacing: '0.1em' }}>
        SECURE BIOMETRIC DATA EXTRACTION PROTOCOL // ARABIC-ENGLISH V1.4
      </footer>
    </div>
  );
};

// Simple JSON highlighter
const renderJSON = (obj: any) => {
  return Object.entries(obj).map(([key, value]) => {
    if (value === null || value === undefined) return null;
    if (typeof value === 'object') {
       return (
         <div key={key} style={{ marginLeft: '1rem' }}>
           <span className="key">"{key}"</span>: {'{'}
           {renderJSON(value)}
           {'}'}
         </div>
       );
    }
    return (
      <div key={key} style={{ marginLeft: '1rem' }}>
        <span className="key">"{key}"</span>: <span className={typeof value}>{JSON.stringify(value)}</span>,
      </div>
    );
  });
};

export default App;
