import { useState } from 'react';
import TextForm from './components/TextForm';
import FileUpload from './components/FileUpload';
import ResultsTable from './components/ResultsTable';
import ConfusionMatrix from './components/ConfusionMatrix';
import './App.css';

function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeView, setActiveView] = useState(null); // "text" or "csv"
  const [confusionMatrixPath, setConfusionMatrixPath] = useState(null);

  const handleTextAnalyze = async (text) => {
    try {
      setLoading(true);
      setError(null);
      
      const res = await fetch('http://localhost:5000/analyze-text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      
      if (!res.ok) {
        throw new Error('API yanıt vermedi');
      }
      
      const data = await res.json();
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      setResults([{
        text: data.text,
        predicted_class: data.results.predicted_class,
        confidence: data.results.confidence,
        probabilities: data.results.probabilities
      }]);
      
      // Confusion matrix yolunu güncelle
      setConfusionMatrixPath(data.confusion_matrix_path);
      
    } catch (err) {
      setError(err.message);
      console.error('Analiz hatası:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (file) => {
    try {
      setLoading(true);
      setError(null);
      
      const formData = new FormData();
      formData.append("file", file);
      
      const res = await fetch('http://localhost:5000/analyze-csv', {
        method: 'POST',
        body: formData
      });
      
      if (!res.ok) {
        throw new Error('API yanıt vermedi');
      }
      
      const data = await res.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Dosya analizi başarısız');
      }
      
      const processedResults = data.data.map(item => ({
        text: item.text,
        predicted_class: item.results.predicted_class,
        confidence: item.results.confidence,
        probabilities: item.results.probabilities
      }));
      
      setResults(processedResults);
    } catch (err) {
      setError(err.message);
      console.error('Dosya yükleme hatası:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1 className="text-center mb-4">Siber Zorbalık Tespiti</h1>

      <div className="info-text">
        📄 <strong>Siber zorbalık tespiti için yorum girin ya da <u>.CSV dosyası</u> yükleyin.</strong>
      </div>

      <div className="button-group">
        <button 
          onClick={() => setActiveView("text")}
          className={activeView === "text" ? "active" : ""}
        >
          Yorum Yaz
        </button>
        <button 
          onClick={() => setActiveView("csv")}
          className={activeView === "csv" ? "active" : ""}
        >
          Dosya Yükle
        </button>
      </div>

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      {activeView === "text" && <TextForm onAnalyze={handleTextAnalyze} disabled={loading} />}
      {activeView === "csv" && <FileUpload onUpload={handleFileUpload} disabled={loading} />}

      {loading ? (
        <div className="loading">
          Analiz yapılıyor...
        </div>
      ) : (
        <>
          <ResultsTable results={results} />
          <ConfusionMatrix imagePath={confusionMatrixPath} />
        </>
      )}

      <img
        src="/HD-wallpaper-anonymus-mask-anonymus-hacker-computer.jpg"
        alt="anonymous hacker"
        className="footer-img"
      />

      <div className="footer-contact">
        <p><strong>Bize Ulaşın:</strong></p>
        <div className="social-links">
          <a href="https://www.instagram.com/jahrein/?hl=tr" target="_blank" rel="noopener noreferrer">
            <i className="fab fa-instagram"></i> Instagram
          </a>
          <a href="http://x.com/jahreinG" target="_blank" rel="noopener noreferrer">
            <i className="fab fa-twitter"></i> Twitter
          </a>
        </div>
      </div>
    </div>
  );
}

export default App;
