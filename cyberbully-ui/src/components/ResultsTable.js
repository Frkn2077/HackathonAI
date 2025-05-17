import React from 'react';

const ResultsTable = ({ results }) => {
  if (!results || results.length === 0) {
    return null;
  }

  return (
    <div className="results-table">
      <h2>Analiz Sonuçları</h2>
      <table>
        <thead>
          <tr>
            <th>Metin</th>
            <th>Tespit Edilen Kategori</th>
            <th>Güven Oranı</th>
            <th>Detaylı Sonuçlar</th>
          </tr>
        </thead>
        <tbody>
          {results.map((result, index) => (
            <tr key={index}>
              <td>{result.text}</td>
              <td>
                <span className={`category ${(result.predicted_class || '').toLowerCase()}`}>
                  {result.predicted_class || 'Bilinmiyor'}
                </span>
              </td>
              <td>{result.confidence ? (result.confidence * 100).toFixed(2) + '%' : 'Bilinmiyor'}</td>
              <td>
                <details>
                  <summary>Tüm Kategoriler</summary>
                  <div className="probabilities">
                    {Object.entries(result.probabilities || {}).map(([category, prob]) => (
                      <div key={category} className="probability-item">
                        <span className="category-name">{category}:</span>
                        <span className="probability-value">{(prob * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </details>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ResultsTable;
