import { useState, useEffect } from 'react';
import { getModelConfig, updateModelConfig } from '../services/api';
import './ModelControl.css';

export default function ModelControl({ mode }) {
  const [config, setConfig] = useState(null);
  const [open, setOpen] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    getModelConfig()
      .then(setConfig)
      .catch((err) => console.warn('Model config load failed:', err));
  }, []);

  if (!config) return null;

  const domain = mode === 'leaf' ? 'leaf' : 'grain';
  const models = config[domain] || {};

  const handleToggle = (model) => {
    const updated = structuredClone(config);
    updated[domain][model].enabled = !updated[domain][model].enabled;
    setConfig(updated);
  };

  const handleWeight = (model, value) => {
    const updated = structuredClone(config);
    updated[domain][model].weight = parseFloat(value);
    setConfig(updated);
  };

  const handleSave = async () => {
    try {
      await updateModelConfig(config);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (e) {
      console.warn('Save failed:', e);
    }
  };

  return (
    <details className="mc" open={open} onToggle={(e) => setOpen(e.target.open)}>
      <summary className="mc-summary">⚙️ Model Settings</summary>
      <div className="mc-body">
        {Object.entries(models).map(([name, cfg]) => (
          <div key={name} className={`mc-row ${!cfg.enabled ? 'mc-disabled' : ''}`}>
            <label className="mc-toggle">
              <input type="checkbox" checked={cfg.enabled} onChange={() => handleToggle(name)} />
              <span className="mc-slider" />
            </label>
            <span className="mc-name">{name.replace(/_/g, ' ')}</span>
            <input type="range" min="0.1" max="5" step="0.1" value={cfg.weight}
              onChange={(e) => handleWeight(name, e.target.value)} disabled={!cfg.enabled} className="mc-range" />
            <span className="mc-val">x{cfg.weight.toFixed(1)}</span>
          </div>
        ))}
        <button className="mc-save" onClick={handleSave}>
          {saved ? '✓ Saved!' : 'Apply'}
        </button>
      </div>
    </details>
  );
}
