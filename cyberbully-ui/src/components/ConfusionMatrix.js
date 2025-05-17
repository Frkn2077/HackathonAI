import React from 'react';

const ConfusionMatrix = ({ imagePath }) => {
    if (!imagePath) {
        return null;
    }

    return (
        <div className="confusion-matrix">
            <h3>Confusion Matrix</h3>
            <div className="matrix-container">
                <img 
                    src={`http://localhost:5000${imagePath}`}
                    alt="Confusion Matrix"
                    onError={(e) => {
                        e.target.style.display = 'none';
                        console.error('Confusion matrix yüklenemedi');
                    }}
                />
            </div>
        </div>
    );
};

export default ConfusionMatrix; 