import React from 'react';

const Scanner = ({ image, isScanning }) => {
  return (
    <div className="scanner-container glass-panel">
      <div className="image-wrapper">
        <img src={image} alt="National ID Scan" className="base-image" />
        {isScanning && <div className="scan-animation"></div>}
        <div className="corner-tl"></div>
        <div className="corner-tr"></div>
        <div className="corner-bl"></div>
        <div className="corner-br"></div>
      </div>

      <style jsx>{`
        .scanner-container {
          padding: 20px;
          position: relative;
          overflow: hidden;
        }
        .image-wrapper {
          position: relative;
          display: flex;
          justify-content: center;
          align-items: center;
        }
        .base-image {
          max-width: 100%;
          border-radius: 8px;
          filter: brightness(0.8) contrast(1.2);
        }
        
        /* Sci-fi corners */
        .corner-tl, .corner-tr, .corner-bl, .corner-br {
          position: absolute;
          width: 20px;
          height: 20px;
          border: 2px solid var(--accent-cyan);
          z-index: 5;
        }
        .corner-tl { top: -5px; left: -5px; border-right: 0; border-bottom: 0; }
        .corner-tr { top: -5px; right: -5px; border-left: 0; border-bottom: 0; }
        .corner-bl { bottom: -5px; left: -5px; border-right: 0; border-top: 0; }
        .corner-br { bottom: -5px; right: -5px; border-left: 0; border-top: 0; }

        .scan-animation {
          position: absolute;
          width: 100%;
          height: 3px;
          background: var(--accent-cyan);
          box-shadow: 0 0 15px var(--accent-cyan), 0 0 5px white;
          animation: scanline 3s infinite ease-in-out;
          z-index: 10;
        }

        @keyframes scanline {
          0% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
      `}</style>
    </div>
  );
};

export default Scanner;
