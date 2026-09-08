import React, { useState, useRef } from 'react';

const UploadZone = ({ onFileSelect, progress, isUploading }) => {
  const [isDragActive, setIsDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileSelect(e.dataTransfer.files[0]);
    }
  };

  return (
    <div 
      className={`glass-panel upload-container ${isDragActive ? 'neon-border active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current.click()}
    >
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={(e) => onFileSelect(e.target.files[0])} 
        style={{ display: 'none' }} 
        accept="image/*"
      />
      
      {!isUploading ? (
        <div className="upload-content">
          <div className="upload-icon">⚡</div>
          <h2 className="neon-text">Initiate Scan</h2>
          <p>Drop identity file or click to browse</p>
          <div className="tech-hint">[ JPEG / PNG / TIFF ]</div>
        </div>
      ) : (
        <div className="upload-progress">
          <div className="progress-info">
            <span className="pulse">Syncing with Node...</span>
            <span className="percentage">{progress}%</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}

      <style jsx>{`
        .upload-container {
          padding: 60px;
          text-align: center;
          cursor: pointer;
          transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
          min-height: 250px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin-bottom: 40px;
        }
        .upload-container.active {
          transform: scale(1.02);
          background: rgba(0, 243, 255, 0.05);
        }
        .upload-icon {
          font-size: 48px;
          margin-bottom: 20px;
        }
        .tech-hint {
          font-size: 10px;
          letter-spacing: 3px;
          margin-top: 20px;
          opacity: 0.5;
        }
        .upload-progress {
          width: 100%;
        }
        .progress-info {
          display: flex;
          justify-content: space-between;
          margin-bottom: 10px;
          font-family: monospace;
          font-size: 14px;
        }
        .progress-bar-container {
          height: 4px;
          background: rgba(255,255,255,0.05);
          width: 100%;
          border-radius: 2px;
          overflow: hidden;
        }
        .progress-bar-fill {
          height: 100%;
          background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
          box-shadow: 0 0 10px var(--accent-cyan);
          transition: width 0.3s ease;
        }
        .pulse {
          animation: pulse 1.5s infinite;
        }
      `}</style>
    </div>
  );
};

export default UploadZone;
