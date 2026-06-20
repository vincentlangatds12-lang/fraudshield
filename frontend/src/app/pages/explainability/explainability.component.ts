import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';

@Component({
  selector: 'app-explainability',
  standalone: true,
  imports: [CommonModule, FormsModule, ChartComponent],
  templateUrl: './explainability.component.html',
  styleUrls: ['./explainability.component.scss'],
})
export class ExplainabilityComponent implements OnInit {
  loading = true;
  activeTab: 'fi' | 'shap' | 'lime' = 'fi';

  // Feature importance
  featureImportance: any[] = [];
  fiChartOpts: any = {};

  // SHAP global
  shapGlobal: any[] = [];
  shapChartOpts: any = {};

  // SHAP / LIME per transaction
  txnId       = '';
  shapLocal:  any  = null;
  limeLocal:  any[] = [];
  shapLocalOpts: any = {};
  limeOpts:      any = {};
  txnLoading  = false;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    Promise.all([
      this.api.getExplainFeatureImportance(25).toPromise(),
      this.api.getShapGlobal(25).toPromise(),
    ]).then(([fi, shap]) => {
      this.featureImportance = fi || [];
      this.shapGlobal        = Array.isArray(shap) ? shap : [];
      this.buildFiChart();
      this.buildShapChart();
      this.loading = false;
    }).catch(() => { this.loading = false; });
  }

  setTab(t: 'fi' | 'shap' | 'lime'): void { this.activeTab = t; }

  loadTxnExplanation(): void {
    const id = parseInt(this.txnId, 10);
    if (!id) return;
    this.txnLoading = true;
    Promise.all([
      this.api.getShapLocal(id).toPromise(),
      this.api.getLimeLocal(id).toPromise(),
    ]).then(([shap, lime]) => {
      this.shapLocal  = shap;
      this.limeLocal  = lime?.explanation || [];
      this.buildShapLocalChart();
      this.buildLimeChart();
      this.txnLoading = false;
    }).catch(() => { this.txnLoading = false; });
  }

  private buildFiChart(): void {
    if (!this.featureImportance.length) return;
    const top = this.featureImportance.slice(0, 20).reverse();
    this.fiChartOpts = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '30%', right: '4%', top: '2%', bottom: '4%' },
      xAxis: { type: 'value', axisLabel: { color: '#64748b' } },
      yAxis: { type: 'category', data: top.map((f: any) => f.feature), axisLabel: { color: '#64748b', fontSize: 11 } },
      series: [{
        type: 'bar', data: top.map((f: any) => f.importance),
        itemStyle: { color: (p: any) => { const v = p.dataIndex / top.length; return `hsl(${220 - v * 60},70%,55%)`; } },
        label: { show: true, position: 'right', fontSize: 10, color: '#94a3b8', formatter: (p: any) => p.value.toFixed(4) },
      }],
      backgroundColor: 'transparent',
    };
  }

  private buildShapChart(): void {
    if (!this.shapGlobal.length) return;
    const top = this.shapGlobal.slice(0, 20).reverse();
    this.shapChartOpts = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '30%', right: '6%', top: '2%', bottom: '4%' },
      xAxis: { type: 'value', axisLabel: { color: '#64748b' }, name: 'mean |SHAP|', nameLocation: 'end' },
      yAxis: { type: 'category', data: top.map((f: any) => f.feature), axisLabel: { color: '#64748b', fontSize: 11 } },
      series: [{
        type: 'bar', data: top.map((f: any) => f.mean_abs_shap),
        itemStyle: { color: '#f59e0b' },
        label: { show: true, position: 'right', fontSize: 10, color: '#94a3b8', formatter: (p: any) => p.value.toFixed(5) },
      }],
      backgroundColor: 'transparent',
    };
  }

  private buildShapLocalChart(): void {
    if (!this.shapLocal?.values) return;
    const entries = Object.entries(this.shapLocal.values as Record<string,number>)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, 15)
      .reverse();
    this.shapLocalOpts = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '32%', right: '6%', top: '4%', bottom: '4%' },
      xAxis: { type: 'value', axisLabel: { color: '#64748b' }, name: 'SHAP value' },
      yAxis: { type: 'category', data: entries.map(e => e[0]), axisLabel: { color: '#64748b', fontSize: 11 } },
      series: [{
        type: 'bar', data: entries.map(e => e[1]),
        itemStyle: { color: (p: any) => (p.value >= 0 ? '#ef4444' : '#10b981') },
        label: { show: true, position: (p: any) => (p.value >= 0 ? 'right' : 'left'), fontSize: 10, color: '#94a3b8', formatter: (p: any) => p.value.toFixed(4) },
      }],
      backgroundColor: 'transparent',
    };
  }

  private buildLimeChart(): void {
    if (!this.limeLocal.length) return;
    const items = [...this.limeLocal].sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight)).slice(0, 15).reverse();
    this.limeOpts = {
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: '35%', right: '6%', top: '4%', bottom: '4%' },
      xAxis: { type: 'value', axisLabel: { color: '#64748b' }, name: 'LIME weight' },
      yAxis: { type: 'category', data: items.map((i: any) => i.feature), axisLabel: { color: '#64748b', fontSize: 10 } },
      series: [{
        type: 'bar', data: items.map((i: any) => i.weight),
        itemStyle: { color: (p: any) => (p.value >= 0 ? '#f97316' : '#06b6d4') },
        label: { show: true, position: (p: any) => (p.value >= 0 ? 'right' : 'left'), fontSize: 10, color: '#94a3b8', formatter: (p: any) => p.value.toFixed(4) },
      }],
      backgroundColor: 'transparent',
    };
  }

  get maxFi(): number { return this.featureImportance[0]?.importance || 1; }
}
