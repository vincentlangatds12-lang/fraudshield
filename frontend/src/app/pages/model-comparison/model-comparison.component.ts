import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';

@Component({
  selector: 'app-model-comparison',
  standalone: true,
  imports: [CommonModule, ChartComponent],
  templateUrl: './model-comparison.component.html',
  styleUrls: ['./model-comparison.component.scss'],
})
export class ModelComparisonComponent implements OnInit {
  loading = true;
  models: any[] = [];
  radarOpts: any = {};
  barOpts:   any = {};
  sortCol  = 'pr_auc';
  sortDir  = -1;

  readonly metrics = [
    { key: 'auc_roc',         label: 'AUC-ROC' },
    { key: 'pr_auc',          label: 'PR-AUC' },
    { key: 'f1_fraud',        label: 'F1 Fraud' },
    { key: 'precision_fraud', label: 'Precision' },
    { key: 'recall_fraud',    label: 'Recall' },
    { key: 'accuracy',        label: 'Accuracy' },
    { key: 'mcc',             label: 'MCC' },
    { key: 'ks_statistic',    label: 'KS-Stat' },
  ];

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.api.getModelComparison().subscribe({
      next: (models) => {
        this.models = models;
        this.sortModels();
        this.buildCharts();
        this.loading = false;
      },
      error: () => { this.loading = false; },
    });
  }

  sortBy(col: string): void {
    if (this.sortCol === col) { this.sortDir *= -1; }
    else { this.sortCol = col; this.sortDir = -1; }
    this.sortModels();
  }

  private sortModels(): void {
    this.models.sort((a, b) => ((b[this.sortCol] ?? 0) - (a[this.sortCol] ?? 0)) * this.sortDir);
  }

  metricClass(val: number): string {
    if (val >= 0.85) return 'metric--excellent';
    if (val >= 0.70) return 'metric--good';
    if (val >= 0.55) return 'metric--fair';
    return 'metric--poor';
  }

  private buildCharts(): void {
    if (!this.models.length) return;
    const names = this.models.map(m => m.classifier_name);

    // Multi-metric bar chart
    const colours = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4','#f97316','#ec4899'];
    this.barOpts = {
      tooltip:  { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend:   { data: this.metrics.map(m => m.label), textStyle: { color: '#94a3b8' }, bottom: 0, type: 'scroll' },
      grid:     { bottom: 60 },
      xAxis:    { type: 'category', data: names, axisLabel: { color: '#64748b', rotate: 20 } },
      yAxis:    { type: 'value', max: 1, axisLabel: { color: '#64748b', formatter: (v: number) => v.toFixed(2) } },
      series:   this.metrics.map((m, i) => ({
        name:       m.label,
        type:       'bar',
        data:       this.models.map(mod => +(mod[m.key] ?? 0).toFixed(4)),
        itemStyle:  { color: colours[i % colours.length] },
        barGap:     '5%',
      })),
      backgroundColor: 'transparent',
    };

    // Radar chart — top 4 models
    const top4 = this.models.slice(0, 4);
    const radarMetrics = ['auc_roc','pr_auc','f1_fraud','mcc','ks_statistic','recall_fraud'];
    const radarLabels  = ['AUC-ROC','PR-AUC','F1','MCC','KS-Stat','Recall'];
    this.radarOpts = {
      tooltip: {},
      legend:  { data: top4.map(m => m.classifier_name), textStyle: { color: '#94a3b8' }, bottom: 0 },
      radar:   {
        indicator: radarLabels.map(name => ({ name, max: 1 })),
        axisName:  { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,.2)' } },
      },
      series: [{
        type: 'radar',
        data: top4.map((m, i) => ({
          name:  m.classifier_name,
          value: radarMetrics.map(k => +(m[k] ?? 0)),
          itemStyle: { color: colours[i] },
          areaStyle: { opacity: .15 },
        })),
      }],
      backgroundColor: 'transparent',
    };
  }
}
