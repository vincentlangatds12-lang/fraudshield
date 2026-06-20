import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';

@Component({
  selector: 'app-model-monitoring',
  standalone: true,
  imports: [CommonModule, ChartComponent],
  templateUrl: './model-monitoring.component.html',
  styleUrls: ['./model-monitoring.component.scss'],
})
export class ModelMonitoringComponent implements OnInit {
  loading     = true;
  runs:  any[] = [];
  versions: any[] = [];
  calibrationOpts: any = {};
  scoreDistOpts:   any = {};
  mlflowUrl = '';

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    Promise.all([
      this.api.getMlflowRuns().toPromise(),
      this.api.getModelVersions().toPromise(),
      this.api.getCalibration().toPromise(),
      this.api.getScoreDistribution().toPromise(),
      this.api.getMlflowUrl().toPromise(),
    ]).then(([runs, versions, cal, score, urlRes]) => {
      this.runs     = runs     || [];
      this.versions = versions || [];
      this.mlflowUrl = urlRes?.url || '';
      this.buildCalibrationChart(cal);
      this.buildScoreChart(score);
      this.loading = false;
    }).catch(() => { this.loading = false; });
  }

  private buildCalibrationChart(cal: any): void {
    if (!cal?.buckets?.length) return;
    this.calibrationOpts = {
      tooltip: { trigger: 'axis' },
      legend: { data: ['Mean Predicted', 'Actual Fraud Rate'], textStyle: { color: '#94a3b8' } },
      xAxis: { type: 'category', data: cal.buckets, axisLabel: { color: '#64748b', rotate: 30, interval: 1 } },
      yAxis: { type: 'value', max: 1, axisLabel: { color: '#64748b', formatter: (v: number) => (v * 100).toFixed(0) + '%' } },
      series: [
        { name: 'Mean Predicted',   type: 'bar',  data: cal.mean_predicted, itemStyle: { color: '#3b82f6', opacity: .75 } },
        { name: 'Actual Fraud Rate',type: 'line', data: cal.actual_rate,    itemStyle: { color: '#ef4444' }, lineStyle: { color: '#ef4444', width: 2 }, symbolSize: 6 },
      ],
      backgroundColor: 'transparent',
    };
  }

  private buildScoreChart(score: any): void {
    if (!score?.bins?.length) return;
    this.scoreDistOpts = {
      tooltip: { trigger: 'axis' },
      xAxis: { type: 'category', data: score.bins, axisLabel: { color: '#64748b', rotate: 30, interval: 3 } },
      yAxis: { type: 'value', axisLabel: { color: '#64748b' } },
      visualMap: {
        show: false, type: 'continuous', min: 0, max: score.bins.length - 1,
        inRange: { color: ['#10b981', '#f59e0b', '#ef4444'] },
      },
      series: [{
        type: 'bar', data: score.counts.map((v: number, i: number) => ({ value: v, itemStyle: {} })),
        colorBy: 'data',
      }],
      backgroundColor: 'transparent',
    };
  }

  metricClass(val: number): string {
    if (val >= 0.85) return 'metric--excellent';
    if (val >= 0.70) return 'metric--good';
    if (val >= 0.55) return 'metric--fair';
    return 'metric--poor';
  }

  formatDuration(ms?: number): string {
    if (!ms) return '—';
    const s = ms / 1000;
    return s < 60 ? s.toFixed(0) + 's' : (s / 60).toFixed(1) + 'm';
  }
}
