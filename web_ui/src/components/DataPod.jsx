import React from 'react';

const DataPod = ({ label, value, confidence, isHeader = false }) => {
  if (!value && !isHeader) return null;

  return (
    <div className={`data-pod ${isHeader ? 'pod-header' : ''}`}>
      <div className="pod-label">{label}</div>
      <div className="pod-value">{value || '---'}</div>
      {confidence && (
        <div className="pod-confidence" style={{ color: confidence > 0.8 ? 'var(--success)' : 'var(--accent-purple)' }}>
          {(confidence * 100).toFixed(0)}% MATCH
        </div>
      )}

      <style jsx>{`
        .data-pod {
          background: rgba(255, 255, 255, 0.03);
          border-left: 2px solid var(--accent-cyan);
          padding: 12px 16px;
          margin-bottom: 12px;
          border-radius: 4px;
          transition: transform 0.2s ease;
        }
        .data-pod:hover {
          transform: translateX(5px);
          background: rgba(255, 255, 255, 0.06);
        }
        .pod-header {
          border-left: 4px solid var(--accent-purple);
          background: rgba(112, 0, 255, 0.05);
          margin-top: 20px;
        }
        .pod-label {
          font-size: 10px;
          text-transform: uppercase;
          letter-spacing: 2px;
          color: var(--text-secondary);
          margin-bottom: 4px;
        }
        .pod-value {
          font-size: 16px;
          font-weight: 500;
          color: var(--text-primary);
        }
        .pod-confidence {
          font-family: monospace;
          font-size: 10px;
          margin-top: 4px;
        }
      `}</style>
    </div>
  );
};

export default DataPod;
