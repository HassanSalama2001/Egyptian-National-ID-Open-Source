import React, { useState, useEffect } from 'react';
import axios from 'axios';
import UploadZone from './components/UploadZone';
import Scanner from './components/Scanner';
import DataPod from './components/DataPod';
import './App.css';

const API_URL = 'http://localhost:8000/ocr'; // Updated to match app.py endpoint

function App() {
  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [data, setData] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [showJson, setShowJson] = useState(false);

  const handleFileSelect = (selectedFile) => {
    if (!selectedFile) return;
    setFile(selectedFile);
    setImagePreview(URL.createObjectURL(selectedFile));
    setData(null);
    uploadFile(selectedFile);
  };

  const uploadFile = async (selectedFile) => {
    setIsUploading(true);
    setProgress(0);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(API_URL, formData, {
        onUploadProgress: (progressEvent) => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProgress(percentCompleted);
        }
      });
      setData(response.data);
    } catch (error) {
      console.error("OCR failed:", error);
      alert("System Error: Failed to process biometric data.");
    } finally {
      setIsUploading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setImagePreview(null);
    setData(null);
    setProgress(0);
    setIsUploading(false);
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo neon-text">EGYPT-ID // HYBRID PIPELINE</div>
        <div className="header-actions">
           <span className="status-indicator">SYSTEM: {isUploading ? 'PROCESSING' : 'IDLE'}</span>
        </div>
      </header>

      <main className="content">
        {!imagePreview ? (
          <div className="entry-point">
            <UploadZone 
              onFileSelect={handleFileSelect} 
              isUploading={isUploading} 
              progress={progress} 
            />
          </div>
        ) : (
          <div className="results-grid">
            <div className="panel image-panel glass-panel">
              <h3 className="panel-title">Original Source Image</h3>
              <Scanner image={imagePreview} isScanning={isUploading} />
              <button className="btn-futuristic reset-btn" onClick={reset} style={{marginTop: '20px'}}>New Session</button>
            </div>
            
            <div className="panel data-panel glass-panel">
              <h3 className="panel-title neon-text">Biometric Field Analysis</h3>
              
              {isUploading ? (
                <div className="loader-container" style={{display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '300px'}}>
                  <div className="cyber-loader"></div>
                  <p className="pulse">Synthesizing Neural Map...</p>
                </div>
              ) : data?.data ? (
                <div className="pods-container">
                  <div className="text-fields">
                    <DataPod label="Legal First Name" value={data.data.front?.first_name} isHeader />
                    <DataPod label="Family / Lineage" value={data.data.front?.full_name} />
                    <DataPod label="Address" value={data.data.front?.address} />
                    <DataPod label="National Identifier" value={data.data.front?.national_id} />
                    <DataPod label="Serial Number" value={data.data.front?.card_serial_number} />
                  </div>

                  <div className="verification-zones">
                     <div className="verification-header">
                        <h4 className="section-subtitle">Visual Context Snippets</h4>
                        <button className="btn-graphics" onClick={() => setShowJson(!showJson)}>
                           {showJson ? 'HIDE RAW JSON' : 'SHOW RAW JSON'}
                        </button>
                     </div>
                     
                     <div className="snippets-grid">
                        {data.data.front?.field_crops && Object.entries(data.data.front.field_crops).map(([field, src]) => (
                          <div key={field} className="snippet-box">
                            <span className="snippet-label">{field.replace('_', ' ')}</span>
                            <img src={src} alt={field} className="field-snippet" />
                          </div>
                        ))}
                     </div>
                  </div>

                  {showJson && (
                    <div className="json-raw" style={{marginTop: '30px', background: 'rgba(0,0,0,0.5)', padding: '15px', borderRadius: '8px', overflow: 'auto', maxHeight: '300px'}}>
                      <pre style={{fontSize: '10px', color: '#0f0'}}>{JSON.stringify(data.data, null, 2)}</pre>
                    </div>
                  )}

                  <div className="meta-info">
                    <span>Node: Localhost // </span>
                    <span>Lat: {data.data.processing_time_ms}ms // </span>
                    <span>Conf: {data.data.confidence * 100}%</span>
                  </div>
                </div>
              ) : (
                <p>Awaiting valid data string...</p>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
