import React, { useEffect, useState } from 'react';
import { TrendingUp, ArrowLeft, AlertTriangle, CheckCircle, Timer } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { TrafficLight } from './TrafficLight';
import { StatsCard } from './StatsCard';

interface TrafficResultsProps {
  signalTime: number;
  vehiclesPerSecond?: number;
  rateOfChange?: number;
  annotatedVideo?: string;
  onReset: () => void;
  vehicleCount?: number;
}

export const TrafficResults: React.FC<TrafficResultsProps> = ({
  signalTime,
  vehiclesPerSecond,
  rateOfChange,
  annotatedVideo,
  onReset,
  vehicleCount
}) => {
  const [timeLeft, setTimeLeft] = useState(signalTime);
  const [currentPhase, setCurrentPhase] = useState<'green' | 'amber' | 'red'>('green');
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    setTimeLeft(signalTime);
    setCurrentPhase('green');
    setIsComplete(false);
  }, [signalTime]);

  useEffect(() => {
    if (timeLeft <= 0) {
      setIsComplete(true);
      return;
    }

    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          setIsComplete(true);
          return 0;
        }
        
        // Phase transitions
        if (prev === 4) {
          setCurrentPhase('amber');
        } else if (prev === 1) {
          setCurrentPhase('red');
        }
        
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, [timeLeft]);

  const progressPercentage = ((signalTime - timeLeft) / signalTime) * 100;

  const handleNewAnalysis = () => {
    onReset();
  };

  return (
    <div className="w-full max-w-6xl mx-auto space-y-6 animate-fade-in-up">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="p-2 bg-success/10 rounded-full">
            <CheckCircle className="w-8 h-8 text-success" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-success bg-clip-text text-transparent">
              Detection Complete
            </h1>
            <p className="text-muted-foreground">
              AI analysis completed successfully
            </p>
          </div>
        </div>
        
        <Button
          variant="outline"
          onClick={handleNewAnalysis}
          className="flex items-center gap-2"
        >
          <ArrowLeft className="w-4 h-4" />
          New Analysis
        </Button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <StatsCard
          title="Rate of Change"
          value={typeof rateOfChange === 'number' ? rateOfChange.toFixed(3) : '—'}
          icon={<TrendingUp className="w-6 h-6" />}
          trend="vehicles/sec²"
          color="primary"
        />
        
        <StatsCard
          title="Optimized Green Time"
          value={`${signalTime}s`}
          icon={<Timer className="w-6 h-6" />}
          trend="based on rate-of-change"
          color="success"
        />

        <StatsCard
          title="Vehicle Count"
          value={typeof vehicleCount === 'number' ? vehicleCount : '—'}
          icon={<CheckCircle className="w-6 h-6" />}
          trend="unique vehicles detected"
          color="warning"
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Traffic Light Control */}
        <Card className="bg-gradient-card border-border shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
            <Timer className="w-5 h-5" />
              Traffic Signal Control
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="flex items-center justify-center">
              <TrafficLight currentPhase={currentPhase} />
            </div>
            
            <div className="text-center space-y-4">
              <div className="text-4xl font-bold text-success">
                {timeLeft > 0 ? timeLeft : 0}
              </div>
              <div className="text-muted-foreground">
                seconds remaining
              </div>
              
              <Progress 
                value={progressPercentage} 
                className="h-3"
              />
              
              <div className="flex items-center justify-center gap-2">
                <Badge 
                  variant={currentPhase === 'green' ? 'default' : 'secondary'}
                  className={`${currentPhase === 'green' ? 'bg-success' : ''}`}
                >
                  {currentPhase.toUpperCase()} PHASE
                </Badge>
                {isComplete && (
                  <Badge variant="outline" className="animate-pulse">
                    CYCLE COMPLETE
                  </Badge>
                )}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Annotated Video */}
        <Card className="bg-gradient-card border-border shadow-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Annotated Detection Video
            </CardTitle>
          </CardHeader>
          <CardContent>
            {annotatedVideo ? (
              <video src={annotatedVideo} className="w-full h-80 object-cover" controls />
            ) : (
              <div className="text-sm text-muted-foreground">Annotated video will appear here after processing.</div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Alert for completion */}
      {isComplete && (
        <Card className="bg-gradient-card border-warning shadow-card animate-pulse">
          <CardContent className="flex items-center gap-4 pt-6">
            <AlertTriangle className="w-8 h-8 text-warning" />
            <div>
              <h3 className="font-semibold text-warning">Signal Cycle Complete</h3>
              <p className="text-muted-foreground">
                Ready for next traffic video analysis
              </p>
            </div>
            <Button
              onClick={handleNewAnalysis}
              className="ml-auto bg-gradient-primary"
            >
              Start New Analysis
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  );
};