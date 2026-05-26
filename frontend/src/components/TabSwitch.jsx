import './TabSwitch.css';

const TABS = [
  { id: 'leaf', label: 'Leaf Analysis', icon: '\u2618\uFE0F' },
  { id: 'grain', label: 'Grain Analysis', icon: '\uD83E\uDE8C' },
];

export default function TabSwitch({ activeTab, onTabChange }) {
  return (
    <div className="tab-switch" role="tablist" aria-label="Analysis mode">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          aria-selected={activeTab === tab.id}
          className={`tab-switch__tab ${activeTab === tab.id ? 'tab-switch__tab--active' : ''}`}
          onClick={() => onTabChange(tab.id)}
        >
          <span className="tab-switch__icon" aria-hidden="true">{tab.icon}</span>
          <span className="tab-switch__label">{tab.label}</span>
        </button>
      ))}
    </div>
  );
}
