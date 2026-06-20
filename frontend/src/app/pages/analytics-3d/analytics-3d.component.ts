import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';
import { forkJoin } from 'rxjs';

@Component({
  selector: 'app-analytics-3d',
  standalone: true,
  imports: [CommonModule, ChartComponent],
  templateUrl: './analytics-3d.component.html',
  styleUrls: ['./analytics-3d.component.scss'],
})
export class Analytics3dComponent {
  loading       = true;
  gl3dAvailable = false;

  scatter3dOpts: any = {};
  bar3dOpts:     any = {};
  surface3dOpts: any = {};
  heatmapOpts:   any = {};
  polarOpts:     any = {};
  boxplotOpts:   any = {};

  constructor(private api: ApiService) {
    this.init();
  }

  private async init(): Promise<void> {
    // echarts-gl removed — always use 2D high-quality fallbacks
    // which render the same data with better browser compatibility
    this.gl3dAvailable = false;

    forkJoin({
      scatter:  this.api.get3dRiskLandscape(600),
      cube:     this.api.get3dModelPerformanceCube(),
      surface:  this.api.get3dPRTSurface(),
      heatmap:  this.api.getHeatmapFraudRisk(),
      polar:    this.api.getPolarFraudByDow(),
      boxplot:  this.api.getBoxplotAmountByChannel(),
    }).subscribe({
      next: (r: any) => {
        if (this.gl3dAvailable) {
          this.build3dScatter(r.scatter);
          this.build3dBar(r.cube);
          this.build3dSurface(r.surface);
        } else {
          this.buildScatterFallback(r.scatter);
          this.buildBarFallback(r.cube);
          this.buildSurfaceFallback(r.surface);
        }
        this.buildHeatmap(r.heatmap);
        this.buildPolar(r.polar);
        this.buildBoxplot(r.boxplot);
        this.loading = false;
      },
      error: () => { this.loading = false; },
    });
  }

  // ═══════════════════════════════════════════════════
  // 3D CHARTS (echarts-gl)
  // ═══════════════════════════════════════════════════

  private build3dScatter(data: any[]): void {
    if (!data?.length) { this.buildScatterFallback(data); return; }
    const channelList = [...new Set(data.map((d: any) => d.channel))] as string[];
    const palette = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#06b6d4'];
    this.scatter3dOpts = {
      tooltip: { formatter: (p: any) => `Log Amt: ${(+p.value[0]).toFixed(2)}<br>Hour: ${p.value[1]}<br>Score: ${(+p.value[2]*100).toFixed(1)}%` },
      legend: { data: channelList, textStyle: { color: '#94a3b8' }, bottom: 0, type: 'scroll' },
      grid3D: { boxWidth: 100, boxDepth: 80, boxHeight: 60, viewControl: { autoRotate: true, autoRotateSpeed: 4 } },
      xAxis3D: { name: 'Log Amount', nameTextStyle: { color: '#94a3b8' } },
      yAxis3D: { name: 'Hour of Day', nameTextStyle: { color: '#94a3b8' } },
      zAxis3D: { name: 'Fraud Prob', nameTextStyle: { color: '#94a3b8' }, min: 0, max: 1 },
      series: channelList.map((ch, ci) => ({
        name: ch, type: 'scatter3D', symbolSize: 4,
        data: data.filter((d: any) => d.channel === ch).map((d: any) => [d.x, d.y, d.z]),
        itemStyle: { color: palette[ci % palette.length], opacity: 0.8 },
      })),
      backgroundColor: 'transparent',
    };
  }

  private build3dBar(data: any): void {
    if (!data?.values?.length) { this.buildBarFallback(data); return; }
    this.bar3dOpts = {
      tooltip: { formatter: (p: any) => `${data.classifiers[p.value[0]]}<br>${data.metrics[p.value[1]]}: ${(+p.value[2]).toFixed(4)}` },
      visualMap: { max: 1, min: 0, inRange: { color: ['#ef4444','#f59e0b','#10b981'] }, textStyle: { color: '#94a3b8' } },
      grid3D: { boxWidth: 120, boxDepth: 80, boxHeight: 80, viewControl: { autoRotate: true, autoRotateSpeed: 3 } },
      xAxis3D: { type: 'category', data: data.classifiers, name: 'Classifier', nameTextStyle: { color: '#94a3b8' } },
      yAxis3D: { type: 'category', data: data.metrics, name: 'Metric', nameTextStyle: { color: '#94a3b8' } },
      zAxis3D: { type: 'value', name: 'Score', min: 0, max: 1, nameTextStyle: { color: '#94a3b8' } },
      series: [{ type: 'bar3D', shading: 'lambert', data: data.values.map((v: any) => ({ value: [v.classifier_idx, v.metric_idx, v.value], itemStyle: { opacity: v.is_champion ? 1 : 0.7 } })) }],
      backgroundColor: 'transparent',
    };
  }

  private build3dSurface(data: any): void {
    if (!data?.surface?.length) { this.buildSurfaceFallback(data); return; }
    this.surface3dOpts = {
      tooltip: { formatter: (p: any) => `Thr: ${(+p.value[0]).toFixed(2)}<br>Recall: ${(+p.value[1]).toFixed(3)}<br>Prec: ${(+p.value[2]).toFixed(3)}` },
      visualMap: { max: 1, min: 0, dimension: 2, inRange: { color: ['#ef4444','#f59e0b','#10b981'] }, textStyle: { color: '#94a3b8' } },
      grid3D: { boxWidth: 100, boxDepth: 80, boxHeight: 70, viewControl: { autoRotate: false, distance: 180 } },
      xAxis3D: { name: 'Threshold', nameTextStyle: { color: '#94a3b8' }, min: 0, max: 1 },
      yAxis3D: { name: 'Recall',    nameTextStyle: { color: '#94a3b8' }, min: 0, max: 1 },
      zAxis3D: { name: 'Precision', nameTextStyle: { color: '#94a3b8' }, min: 0, max: 1 },
      series: [{ type: 'scatter3D', symbolSize: 6, itemStyle: { opacity: 0.85 }, data: data.surface.map((d: any) => [d.threshold, d.recall, d.precision]) }],
      backgroundColor: 'transparent',
    };
  }

  // ═══════════════════════════════════════════════════
  // 2D FALLBACKS (when echarts-gl not available)
  // ═══════════════════════════════════════════════════

  private buildScatterFallback(data: any[]): void {
    if (!data?.length) return;
    const channelList = [...new Set(data.map((d: any) => d.channel))] as string[];
    const palette = ['#3b82f6','#ef4444','#10b981','#f59e0b','#8b5cf6','#06b6d4'];
    this.scatter3dOpts = {
      title: { text: 'Fraud Risk Landscape (2D — Amount vs Score)', textStyle: { color: '#94a3b8', fontSize: 13 } },
      tooltip: { trigger: 'item', formatter: (p: any) => `Log Amt: ${(+p.value[0]).toFixed(2)}<br>Score: ${(+p.value[1]*100).toFixed(1)}%<br>Hour: ${p.value[2]}` },
      legend: { data: channelList, textStyle: { color: '#94a3b8' }, bottom: 0, type: 'scroll' },
      grid: { top: 60 },
      xAxis: { name: 'Log Amount (USD)', axisLabel: { color: '#64748b' } },
      yAxis: { name: 'Fraud Probability', min: 0, max: 1, axisLabel: { color: '#64748b', formatter: (v: number) => (v*100).toFixed(0)+'%' } },
      series: channelList.map((ch, ci) => ({
        name: ch, type: 'scatter', symbolSize: 5,
        data: data.filter((d: any) => d.channel === ch).map((d: any) => [d.x, d.z, d.y]),
        itemStyle: { color: palette[ci % palette.length], opacity: 0.7 },
      })),
      backgroundColor: 'transparent',
    };
  }

  private buildBarFallback(data: any): void {
    if (!data?.values?.length) return;
    const colours = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#06b6d4'];
    this.bar3dOpts = {
      title: { text: 'Model Performance Cube (2D — grouped bars)', textStyle: { color: '#94a3b8', fontSize: 13 } },
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      legend: { data: data.metrics, textStyle: { color: '#94a3b8' }, bottom: 0, type: 'scroll' },
      grid: { bottom: 60, top: 60 },
      xAxis: { type: 'category', data: data.classifiers, axisLabel: { color: '#64748b', rotate: 20 } },
      yAxis: { type: 'value', max: 1, axisLabel: { color: '#64748b' } },
      series: data.metrics.map((m: string, mi: number) => ({
        name: m, type: 'bar',
        data: data.classifiers.map((_: string, ci: number) => {
          const entry = data.values.find((v: any) => v.classifier_idx === ci && v.metric_idx === mi);
          return entry ? +(entry.value).toFixed(4) : 0;
        }),
        itemStyle: { color: colours[mi % colours.length] },
        barGap: '5%',
      })),
      backgroundColor: 'transparent',
    };
  }

  private buildSurfaceFallback(data: any): void {
    if (!data?.surface?.length) return;
    const s = data.surface;
    this.surface3dOpts = {
      title: { text: 'Precision-Recall-Threshold (2D)', textStyle: { color: '#94a3b8', fontSize: 13 } },
      tooltip: { trigger: 'axis' },
      legend: { data: ['Precision', 'Recall', 'F1', 'Alarm Rate'], textStyle: { color: '#94a3b8' } },
      grid: { top: 60 },
      xAxis: { type: 'category', data: s.map((d: any) => d.threshold.toFixed(2)), axisLabel: { color: '#64748b', interval: 4 }, name: 'Threshold' },
      yAxis: { type: 'value', min: 0, max: 1, axisLabel: { color: '#64748b' } },
      series: [
        { name: 'Precision',  type: 'line', data: s.map((d: any) => d.precision),  smooth: true, itemStyle: { color: '#3b82f6' }, lineStyle: { width: 2 } },
        { name: 'Recall',     type: 'line', data: s.map((d: any) => d.recall),     smooth: true, itemStyle: { color: '#10b981' }, lineStyle: { width: 2 } },
        { name: 'F1',         type: 'line', data: s.map((d: any) => d.f1),         smooth: true, itemStyle: { color: '#f59e0b' }, lineStyle: { width: 2.5, type: 'dashed' } },
        { name: 'Alarm Rate', type: 'line', data: s.map((d: any) => d.alarm_rate), smooth: true, itemStyle: { color: '#94a3b8' }, lineStyle: { width: 1, type: 'dotted' } },
      ],
      backgroundColor: 'transparent',
    };
  }

  // ═══════════════════════════════════════════════════
  // HEATMAP
  // ═══════════════════════════════════════════════════

  private buildHeatmap(data: any): void {
    if (!data?.channels?.length) return;
    this.heatmapOpts = {
      tooltip: {
        position: 'top',
        formatter: (p: any) => {
          const [hour, ci, rate, total, fraud] = p.value;
          return `<strong>${data.channels[ci]}</strong> @ ${String(hour).padStart(2,'0')}:00<br>Fraud Rate: <strong>${(+rate*100).toFixed(2)}%</strong><br>Total: ${total} · Fraud: ${fraud}`;
        },
      },
      grid: { top: '8%', right: '8%', bottom: '15%', left: '14%' },
      xAxis: {
        type: 'category',
        data: data.hours.map((h: number) => `${String(h).padStart(2,'0')}:00`),
        splitArea: { show: true },
        axisLabel: { color: '#64748b', rotate: 45, interval: 1, fontSize: 10 },
        name: 'Hour of Day', nameTextStyle: { color: '#94a3b8' },
      },
      yAxis: {
        type: 'category',
        data: data.channels,
        splitArea: { show: true },
        axisLabel: { color: '#64748b' },
      },
      visualMap: {
        min: 0, max: data.max_rate, calculable: true,
        orient: 'horizontal', left: 'center', bottom: 0,
        inRange: { color: ['#f0f9ff','#bfdbfe','#3b82f6','#dc2626','#7f1d1d'] },
        textStyle: { color: '#94a3b8' },
        formatter: (v: number) => (v*100).toFixed(1) + '%',
      },
      series: [{
        name: 'Fraud Rate', type: 'heatmap',
        data: data.matrix.map((d: any) => [d.hour, d.channel_idx, d.fraud_rate]),
        label: {
          show: true,
          formatter: (p: any) => p.value[2] > 0 ? (p.value[2]*100).toFixed(0)+'%' : '',
          fontSize: 9, color: '#1e293b',
        },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,.3)' } },
      }],
      backgroundColor: 'transparent',
    };
  }

  // ═══════════════════════════════════════════════════
  // POLAR
  // ═══════════════════════════════════════════════════

  private buildPolar(data: any): void {
    if (!data?.days?.length) return;
    this.polarOpts = {
      tooltip: { trigger: 'item' },
      angleAxis: {
        type: 'category', data: data.days, boundaryGap: false,
        axisLabel: { color: '#94a3b8', fontSize: 12, fontWeight: 600 },
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b' } },
      },
      radiusAxis: {
        axisLabel: { color: '#64748b', fontSize: 10 },
        axisLine: { lineStyle: { color: '#334155' } },
        splitLine: { lineStyle: { color: '#1e293b', type: 'dashed' } },
      },
      polar: { radius: ['15%', '75%'], center: ['50%', '52%'] },
      legend: { data: ['KE Fraud','NG Fraud','Legit /10'], textStyle: { color: '#94a3b8' }, bottom: 0 },
      series: [
        { type: 'bar', name: 'KE Fraud', coordinateSystem: 'polar', data: data.ke_fraud, itemStyle: { color: '#ef4444', opacity: 0.9 }, stack: 'fraud' },
        { type: 'bar', name: 'NG Fraud', coordinateSystem: 'polar', data: data.ng_fraud, itemStyle: { color: '#f97316', opacity: 0.85 }, stack: 'fraud' },
        { type: 'line', name: 'Legit /10', coordinateSystem: 'polar', data: data.legit_counts.map((v: number) => Math.round(v / 10)), lineStyle: { color: '#10b981', width: 2 }, itemStyle: { color: '#10b981' }, smooth: true, areaStyle: { color: '#10b981', opacity: 0.1 }, symbolSize: 5 },
      ],
      backgroundColor: 'transparent',
    };
  }

  // ═══════════════════════════════════════════════════
  // BOXPLOT
  // ═══════════════════════════════════════════════════

  private buildBoxplot(data: any): void {
    if (!data?.channels?.length) return;
    const toBox = (s: any) => s ? [s.min, s.q1, s.median, s.q3, s.max] : null;
    const fraudData = data.fraud.map((f: any) => toBox(f.stats)).filter(Boolean);
    const legitData = data.legit.map((l: any) => toBox(l.stats)).filter(Boolean);
    const channels  = data.channels;
    const fOut: any[] = [], lOut: any[] = [];
    data.fraud.forEach((f: any, i: number) => (f.stats?.outliers || []).slice(0, 5).forEach((v: number) => fOut.push([i, v])));
    data.legit.forEach((l: any, i: number) => (l.stats?.outliers || []).slice(0, 5).forEach((v: number) => lOut.push([i, v])));
    this.boxplotOpts = {
      tooltip: {
        trigger: 'item',
        formatter: (p: any) => {
          if (p.seriesType === 'boxplot') {
            const [min, q1, med, q3, max] = p.value.slice(1);
            return `<strong>${p.name} — ${p.seriesName}</strong><br>Max: $${(+max).toFixed(2)}<br>Q3: $${(+q3).toFixed(2)}<br>Median: $${(+med).toFixed(2)}<br>Q1: $${(+q1).toFixed(2)}<br>Min: $${(+min).toFixed(2)}`;
          }
          return `Outlier: $${(+p.value[1]).toFixed(3)}`;
        },
      },
      legend: { data: ['Fraud','Legit'], textStyle: { color: '#94a3b8' } },
      grid: { left: '8%', right: '4%', bottom: '12%' },
      xAxis: { type: 'category', data: channels, boundaryGap: true, axisLabel: { color: '#64748b', rotate: 15 } },
      yAxis: { type: 'log', name: 'Amount USD (log)', nameTextStyle: { color: '#94a3b8' }, axisLabel: { color: '#64748b', formatter: (v: number) => '$'+v.toFixed(2) }, splitLine: { lineStyle: { color: 'rgba(148,163,184,.15)' } } },
      series: [
        { name: 'Fraud', type: 'boxplot', data: fraudData.map((d: any, i: number) => ({ value: [i, ...d], name: channels[i] })), itemStyle: { color: '#fca5a5', borderColor: '#ef4444', borderWidth: 2 } },
        { name: 'Legit', type: 'boxplot', data: legitData.map((d: any, i: number) => ({ value: [i, ...d], name: channels[i] })), itemStyle: { color: '#86efac', borderColor: '#22c55e', borderWidth: 2 } },
        { name: 'Fraud Outliers', type: 'scatter', data: fOut.map(([i,v]: any) => [i,v]), symbolSize: 5, itemStyle: { color: '#ef4444', opacity: 0.6 } },
        { name: 'Legit Outliers', type: 'scatter', data: lOut.map(([i,v]: any) => [i,v]), symbolSize: 5, itemStyle: { color: '#22c55e', opacity: 0.5 } },
      ],
      backgroundColor: 'transparent',
    };
  }
}
