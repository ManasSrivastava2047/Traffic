import React, { useState } from 'react';

const regions = [
  'North West Delhi',
  'North East Delhi',
  'South West Delhi',
  'South East Delhi'
];

const intersections = [
  'Sai Baba Chowk',
  'Madhuban Chowk',
  'Rithala Metro Junction',
  'Deepali Chowk',
  'Mukarba Chowk',
  'Punjabi Bagh Crossing'
];

export const DriverView: React.FC = () => {
  const [region, setRegion] = useState(regions[0]);
  const [intersectionName, setIntersectionName] = useState(intersections[0]);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [latest, setLatest] = useState<any | null>(null);

  const handleConfirm = async () => {
    if (!region || !intersectionName) {
      setError('Please select both Region and Intersection Name.');
      return;
    }
    setError('');
    setLoading(true);
    try {
      // fetch latest stored row
      const params = new URLSearchParams();
      params.append('region_name', region);
      params.append('intersection_id', intersectionName);
      const resp = await fetch(`http://localhost:5000/api/db/latest?${params.toString()}`);
      const text = await resp.text();
      let data: any = null;
      try {
        data = JSON.parse(text);
      } catch (err) {
        data = { error: text };
      }
      if (!resp.ok) {
        setError(data.error || 'Failed to fetch latest data');
        setLatest(null);
      } else {
        setLatest(data.row || null);
      }
    } catch (err: any) {
      setError(err.message || String(err));
      setLatest(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-bg flex items-start justify-center py-12">
      <div className="w-full max-w-2xl">
        <div className="space-y-6">
          <div className="p-6 rounded-xl bg-card border border-border">
            <h2 className="text-xl font-semibold">Driver — Check Intersection</h2>
            <div className="mt-4 space-y-3">
              <select
                aria-label="Select Region"
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white"
              >
                {regions.map((r) => (
                  <option key={r} value={r} className="bg-card">
                    {r}
                  </option>
                ))}
              </select>
              <select
                aria-label="Select Intersection"
                value={intersectionName}
                onChange={(e) => setIntersectionName(e.target.value)}
                className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white"
              >
                {intersections.map((intersection) => (
                  <option key={intersection} value={intersection} className="bg-card">
                    {intersection}
                  </option>
                ))}
              </select>
              {error && <div className="text-red-400">{error}</div>}
              <div className="flex gap-3 mt-2">
                <button onClick={handleConfirm} className="bg-primary text-white px-4 py-2 rounded">Confirm Intersection</button>
              </div>
            </div>
          </div>

          <div>
            {loading && <div className="text-sm text-muted-foreground">Loading...</div>}
            {latest && (
              <div className="p-4 rounded-xl bg-card border border-border">
                <h3 className="font-semibold mb-4">Traffic Status for {latest.Region_Name} / {latest.Intersection_ID}</h3>
                {(() => {
                    const mg = parseInt(String(latest.Max_Green_Time || '0'), 10) || 0;
                    if (mg > 55) {
                      return (
                        <div className="mt-3 p-3 bg-red-500/10 border border-red-300 rounded">
                          <div className="font-semibold text-red-600">Traffic density: Very High</div>
                          <div>Expect significant delays. Consider alternate routes or avoid the area if possible.</div>
                          <div className="text-sm text-muted-foreground">Emergency and priority vehicles may be given extended green time.</div>
                        </div>
                      );
                    }
                    if (mg < 30) {
                      return (
                        <div className="mt-3 p-3 bg-green-500/10 border border-green-300 rounded">
                          <div className="font-semibold text-green-700">Traffic density: Low</div>
                          <div>Traffic is flowing smoothly. Minimal delays expected.</div>
                          <div className="text-sm text-muted-foreground">You should be able to cross quickly. Drive safely and follow signals.</div>
                        </div>
                      );
                    }
                    return (
                      <div className="mt-3 p-3 bg-yellow-500/10 border border-yellow-300 rounded">
                        <div className="font-semibold text-yellow-800">Traffic density: Moderate</div>
                        <div>Some delays possible. Exercise caution and follow traffic directions.</div>
                        <div className="text-sm text-muted-foreground">Peak adjustments may be active; expect variable signal timings.</div>
                      </div>
                    );
                  })()}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};