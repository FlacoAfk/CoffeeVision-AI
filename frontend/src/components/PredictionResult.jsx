import './PredictionResult.css';
import ModelControl from './ModelControl';

function formatClassName(className) {
  if (!className) return 'Unknown';
  let formatted = className.replace(/([a-z])([A-Z])/g, '$1 $2');
  formatted = formatted.replace(/-/g, ' ');
  if (formatted === 'Danado') formatted = 'Dañado';
  return formatted;
}

function isHealthy(className) {
  if (!className) return false;
  const lower = className.toLowerCase();
  return lower === 'sanas' || lower === 'sano';
}

export default function PredictionResult({ prediction, mode, imageUrl }) {
  if (!prediction) return null;

  const { predicted_class, confidence = 0, top_3 = [], individual = [], processing_time_ms } = prediction;
  const healthy = isHealthy(predicted_class);
  const displayName = formatClassName(predicted_class);
  const pct = Math.round(confidence * 100);

  // Build gauge SVG path
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;
  const barColor = confidence >= 0.9 ? 'var(--color-success)' : confidence >= 0.7 ? 'var(--color-warning)' : 'var(--color-danger)';

  return (
    <div className={`prediction-result animate-scale-in ${healthy ? 'prediction-result--healthy' : 'prediction-result--issue'}`}>
      
      {/* Image with overlay */}
      {imageUrl && (
        <div className="pr-image-wrap">
          <img src={imageUrl} alt="Analyzed" className="pr-image" />
          <div className={`pr-overlay ${healthy ? 'overlay-ok' : 'overlay-bad'}`}>
            <span className="pr-overlay-badge">{healthy ? 'Healthy' : 'Issue Detected'}</span>
            <span className="pr-overlay-class">{displayName}</span>
            <span className="pr-overlay-pct">{pct}% confidence</span>
          </div>
        </div>
      )}

      {/* Gauge + result */}
      <div className="pr-hero">
        <svg viewBox="0 0 120 120" className="pr-gauge">
          <circle cx="60" cy="60" r={radius} fill="none" stroke="var(--color-neutral-200)" strokeWidth="8" />
          <circle cx="60" cy="60" r={radius} fill="none" stroke={barColor} strokeWidth="8"
            strokeLinecap="round" strokeDasharray={circumference} strokeDashoffset={offset}
            transform="rotate(-90 60 60)" />
          <text x="60" y="55" textAnchor="middle" fontSize="22" fontWeight="700" fill={barColor}>{pct}%</text>
          <text x="60" y="75" textAnchor="middle" fontSize="11" fill="var(--color-neutral-500)">{displayName}</text>
        </svg>
        <div className="pr-hero-text">
          <span className={`badge ${healthy ? 'badge-success' : 'badge-danger'}`}>
            {healthy ? 'Healthy' : 'Issue Detected'}
          </span>
          <h3 className="pr-class">{displayName}</h3>
          <p className="pr-subtitle">Ensemble of {individual.length} models</p>
        </div>
      </div>

      {/* Top 3 */}
      {top_3.length > 0 && (
        <div className="pr-top3">
          <h4 className="pr-section-title">Top Predictions</h4>
          {top_3.map((item, i) => {
            const ip = Math.round(item.confidence * 100);
            return (
              <div key={item.class} className="pr-top3-row">
                <span className="pr-top3-rank">{i + 1}</span>
                <div className="pr-top3-info">
                  <span className="pr-top3-name">{formatClassName(item.class)}</span>
                  <div className="progress-bar"><div className="progress-bar-fill" style={{ width: `${ip}%`, backgroundColor: i === 0 ? barColor : 'var(--color-neutral-300)' }} /></div>
                </div>
                <span className="pr-top3-pct">{ip}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Individual models */}
      {individual.length > 0 && (
        <div className="pr-models">
          <h4 className="pr-section-title">Model Contributions <span className="pr-count">{individual.length} models</span></h4>
          {individual.map((m) => {
            const mp = Math.round((m.confidence || 0) * 100);
            const w = (m.weight || 0) * 100;
            return (
              <div key={m.model} className="pr-model-row">
                <div className="pr-model-header">
                  <span className="pr-model-name">{m.model.replace(/_/g, ' ')}</span>
                  <span className="pr-model-weight">x{w.toFixed(0)}%</span>
                </div>
                <div className="pr-model-bar">
                  <div className="pr-model-fill" style={{ width: `${mp}%`, backgroundColor: getColor(m.confidence || 0) }} />
                  <span className="pr-model-val">{mp}% {formatClassName(m.predicted_class)}</span>
                </div>
              </div>
            );
          })}
          {/* Ensemble verdict */}
          <div className={`pr-verdict ${healthy ? 'verdict-ok' : 'verdict-bad'}`}>
            <span className="pr-verdict-label">ENSEMBLE</span>
            <span className="pr-verdict-class">{displayName}</span>
            <span className="pr-verdict-pct">{pct}%</span>
          </div>
        </div>
      )}

      <ModelControl mode={mode} />

      {processing_time_ms != null && (
        <p className="pr-meta">Processed in {Math.round(processing_time_ms)} ms</p>
      )}
    </div>
  );
}

function getColor(c) {
  if (c >= 0.9) return 'var(--color-success)';
  if (c >= 0.7) return 'var(--color-warning)';
  return 'var(--color-danger)';
}
