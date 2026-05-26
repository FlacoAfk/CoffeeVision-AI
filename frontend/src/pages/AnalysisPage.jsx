import { useState } from 'react';
import TabSwitch from '../components/TabSwitch';
import ImageUpload from '../components/ImageUpload';
import PredictionResult from '../components/PredictionResult';
import './AnalysisPage.css';

export default function AnalysisPage() {
  const [activeTab, setActiveTab] = useState('leaf');
  const [prediction, setPrediction] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [error, setError] = useState(null);

  const handleResult = (result, previewUrl) => {
    setPrediction(result);
    setImageUrl(previewUrl);
    setError(null);
  };

  const handleError = (msg) => {
    setError(msg);
    setPrediction(null);
  };

  const handleReset = () => {
    setPrediction(null);
    setError(null);
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="analysis-page">
      <header className="app-header">
        <h1 className="app-header__title">CoffeeVision AI</h1>
        <p className="app-header__subtitle">Coffee leaf disease & grain quality classification</p>
      </header>

      <main className="container">
        <TabSwitch activeTab={activeTab} onTabChange={handleTabChange} />

        <div className="card">
          {!prediction ? (
            <ImageUpload
              mode={activeTab}
              onResult={handleResult}
              onError={handleError}
            />
          ) : (
            <>
              <PredictionResult prediction={prediction} mode={activeTab} imageUrl={imageUrl} />
              <button
                className="btn btn-outline analysis-page__reset"
                onClick={handleReset}
                type="button"
              >
                Analyze Another Image
              </button>
            </>
          )}

          {error && !prediction && (
            <div className="analysis-page__error" role="alert">
              {error}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
