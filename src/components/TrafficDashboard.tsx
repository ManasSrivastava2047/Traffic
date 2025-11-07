import React, { useState, useEffect } from 'react';
import { Activity, Zap } from 'lucide-react';
import { TrafficUpload } from './TrafficUpload';
import { TrafficResults } from './TrafficResults';
import { TrafficResultsMulti } from './TrafficResultsMulti';
import { SystemStatus } from './SystemStatus';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

interface LaneResult {
  laneId: number;
  signalTime: number;
  vehiclesPerSecond?: number;
  rateOfChange?: number;
  annotatedVideo?: string;
  vehicleCount?: number;
  direction?: string;
}

interface DetectionResult {
  signalTime: number;
  vehiclesPerSecond?: number;
  rateOfChange?: number;
  annotatedVideo?: string;
  vehicleCount?: number;
  lanes?: LaneResult[];
}

type AppState = 'setup' | 'upload' | 'processing' | 'results';

export const TrafficDashboard: React.FC = () => {
  // ===== State Variables =====
  const [appState, setAppState] = useState<AppState>('setup');
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState<DetectionResult | null>(null);
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

  const [region, setRegion] = useState(regions[0]);
  const [intersectionName, setIntersectionName] = useState(intersections[0]);
  const [intersectionConfirmed, setIntersectionConfirmed] = useState(false);
  const [error, setError] = useState('');

  // Live clock
  const formatTime = (date: Date): string => {
    const pad = (n: number) => String(n).padStart(2, '0');
    return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
  };
  const [currentTime, setCurrentTime] = useState<string>(formatTime(new Date()));
  useEffect(() => {
    const intervalId = setInterval(() => setCurrentTime(formatTime(new Date())), 1000);
    return () => clearInterval(intervalId);
  }, []);

  // ===== Helper Functions =====
  const computeOptimizedGreenTime = (baseSeconds: number, rateOfChange?: number): number => {
    if (rateOfChange === undefined || rateOfChange === null || isNaN(rateOfChange)) return baseSeconds;
    const K = 10;
    const unclamped = baseSeconds + rateOfChange * K;
    return Math.round(Math.max(15, Math.min(65, unclamped)));
  };

  const simulateAIProcessing = (video: File | null): Promise<DetectionResult> =>
    new Promise(async (resolve, reject) => {
      try {
        setProgress(20);
        if (!video) throw new Error('No video provided');
        const formData = new FormData();
        formData.append('video', video);
        const resp = await fetch('http://localhost:5000/api/video/analyze', { method: 'POST', body: formData });
        if (!resp.ok) throw new Error('Video analysis failed');
        const data = await resp.json();
        setProgress(100);
        resolve({
          vehiclesPerSecond: data.vehiclesPerSecond,
          rateOfChange: data.rateOfChange,
          signalTime: computeOptimizedGreenTime(35, data.rateOfChange),
          annotatedVideo: data.annotatedVideo ? `http://localhost:5000${data.annotatedVideo}` : undefined,
          vehicleCount: data.vehicleCount,
        });
      } catch (err) {
        reject(err);
      }
    });

  const simulateAIProcessingMulti = (
    lanes: { north?: File | null; south?: File | null; east?: File | null; west?: File | null; lane1?: File | null; lane2?: File | null; lane3?: File | null; lane4?: File | null }
  ): Promise<DetectionResult> =>
    new Promise(async (resolve, reject) => {
      try {
        setProgress(20);
        const vForm = new FormData();
        (['north','south','east','west','lane1','lane2','lane3','lane4'] as const).forEach(k => {
          const f = lanes[k];
          if (f) vForm.append(k, f);
        });
        const resp = await fetch('http://localhost:5000/api/video/analyze-multi', { method: 'POST', body: vForm });
        if (!resp.ok) throw new Error('Multi-lane analysis failed');
        const data = await resp.json();
        setProgress(100);
        const lanesOut: LaneResult[] = (data.lanes || []).map((ln: any, idx: number) => ({
          laneId: typeof ln.laneId === 'number' ? ln.laneId : idx + 1,
          signalTime: ln.signalTime,
          vehiclesPerSecond: ln.vehiclesPerSecond,
          rateOfChange: ln.rateOfChange,
          annotatedVideo: ln.annotatedVideo ? `http://localhost:5000${ln.annotatedVideo}` : undefined,
          vehicleCount: ln.vehicleCount,
          direction: ln.direction,
        }));
        resolve({ signalTime: 0, lanes: lanesOut });
      } catch (err) {
        reject(err);
      }
    });

  const handleFileUpload = async (
    video: File | null,
    multi?: { north?: File | null; south?: File | null; east?: File | null; west?: File | null; lane1?: File | null; lane2?: File | null; lane3?: File | null; lane4?: File | null }
  ) => {
    setAppState('processing');
    setProgress(0);
    try {
      const result = multi &&
        (multi.north || multi.south || multi.east || multi.west || multi.lane1 || multi.lane2 || multi.lane3 || multi.lane4)
        ? await simulateAIProcessingMulti(multi)
        : await simulateAIProcessing(video);
      setResults(result);
      setAppState('results');
    } catch (err) {
      console.error('Processing failed:', err);
      setAppState('upload');
    }
  };

  const handleReset = () => {
    setAppState('setup');
    setProgress(0);
    setResults(null);
    setRegion('');
    setIntersectionName(intersections[0]);
    setIntersectionConfirmed(false);
    setError('');
  };

  const handleConfirmIntersection = async () => {
    if (!region || !intersectionName) {
      setError('Please select both Region and Intersection Name.');
      return;
    }
    setError('');
    try {
      // Post metadata to backend so server can use it during analyze
      const resp = await fetch('http://localhost:5000/api/metadata', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ region_name: region, intersection_id: intersectionName }),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Failed to set metadata: ${txt}`);
      }
      setIntersectionConfirmed(true);
      setAppState('upload');
    } catch (err: any) {
      console.error('Failed to confirm intersection:', err);
      setError('Failed to confirm intersection on server. See console.');
    }
  };

  // ===== Render =====
  return (
    <div className="min-h-screen bg-gradient-bg">
      {/* Header */}
      <div className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-primary/10 rounded-lg">
              <Activity className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                AI Traffic Management
              </h1>
              <p className="text-sm text-muted-foreground">Real-time traffic optimization system</p>
            </div>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span className="font-mono tabular-nums text-muted-foreground text-base min-w-[88px] text-right">{currentTime}</span>
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-success rounded-full animate-pulse" />
              <span className="text-success font-medium">System Online</span>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8 grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Main Content */}
        <div className="xl:col-span-3 space-y-6">
          {/* Setup Form */}
          {!intersectionConfirmed && (
            <div className="space-y-6">
              <Card className="bg-gradient-card border-border shadow-card">
                <CardHeader>
                  <CardTitle>Enter Region</CardTitle>
                </CardHeader>
                <CardContent>
                  <select
                    value={region}
                    onChange={(e) => setRegion(e.target.value)}
                    className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-inner transition-all duration-200"
                    aria-label="Select Region"
                  >
                    {regions.map((r) => (
                      <option key={r} value={r} className="bg-card">
                        {r}
                      </option>
                    ))}
                  </select>
                </CardContent>
              </Card>

              <Card className="bg-gradient-card border-border shadow-card">
                <CardHeader>
                  <CardTitle>Select Intersection</CardTitle>
                </CardHeader>
                <CardContent>
                  <select
                    value={intersectionName}
                    onChange={(e) => setIntersectionName(e.target.value)}
                    className="w-full p-3 rounded-xl border border-gray-300 bg-transparent text-white focus:outline-none focus:ring-2 focus:ring-indigo-400 shadow-inner transition-all duration-200"
                    aria-label="Select Intersection"
                  >
                    {intersections.map((intersection) => (
                      <option key={intersection} value={intersection} className="bg-card">
                        {intersection}
                      </option>
                    ))}
                  </select>
                </CardContent>
              </Card>

              {error && <div className="text-red-500 text-sm">{error}</div>}

              <button
                onClick={handleConfirmIntersection}
                className="bg-primary text-white px-4 py-2 rounded"
              >
                Confirm Intersection
              </button>
            </div>
          )}

          {/* Upload Section */}
          {intersectionConfirmed && (appState === 'upload' || appState === 'processing') && (
            <TrafficUpload onUpload={handleFileUpload} isProcessing={appState === 'processing'} progress={progress} />
          )}

          {/* Results Section */}
          {appState === 'results' && results && (
            results.lanes && results.lanes.length > 0 ? (
              <TrafficResultsMulti lanes={results.lanes} onReset={handleReset} />
            ) : (
              <TrafficResults
                signalTime={results.signalTime}
                vehiclesPerSecond={results.vehiclesPerSecond}
                rateOfChange={results.rateOfChange}
                annotatedVideo={results.annotatedVideo}
                vehicleCount={results.vehicleCount}
                onReset={handleReset}
              />
            )
          )}
        </div>

        {/* Sidebar */}
        <div className="xl:col-span-1 space-y-6">
          <SystemStatus />
          {/* Recent Activity */}
          <Card className="bg-gradient-card border-border shadow-card">
            <CardHeader>
              <CardTitle className="text-lg">Recent Activity</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="text-sm">
                <div className="flex justify-between items-center mb-1">
                  <span>Highway Junction A</span>
                  <span className="text-xs text-muted-foreground">2min ago</span>
                </div>
                <div className="text-xs text-muted-foreground">15 vehicles detected • 45s signal</div>
              </div>
              <div className="text-sm">
                <div className="flex justify-between items-center mb-1">
                  <span>Main Street Cross</span>
                  <span className="text-xs text-muted-foreground">5min ago</span>
                </div>
                <div className="text-xs text-muted-foreground">8 vehicles detected • 32s signal</div>
              </div>
              <div className="text-sm">
                <div className="flex justify-between items-center mb-1">
                  <span>Park Avenue</span>
                  <span className="text-xs text-muted-foreground">12min ago</span>
                </div>
                <div className="text-xs text-muted-foreground">22 vehicles detected • 60s signal</div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};
