import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import { forkJoin, of } from 'rxjs';
import { catchError } from 'rxjs/operators';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [CommonModule, RouterLink, ChartComponent],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
})
export class DashboardComponent implements OnInit {
  loading = true;
  summary: any = {};
  fraudByChannel: any[] = [];
  fraudByCountry: any[] = [];
  amountDist: any = {};
  dailyTrend: any = {};
  scoreDistribution: any = {};
  imbalanceReport: any = {};
  topFlagged: any[] = [];
  prCurve: any = {};
  reviewStats: any = {};
  modelComparison: any[] = [];   // ← new

  // Hero metric
  recallPct        = '—';
  recallDashOffset = 314;

  // Chart options
  channelChartOpts: any = {};
  countryChartOpts: any = {};
  amountChartOpts:  any = {};
  trendChartOpts:   any = {};
  scoreChartOpts:   any = {};
  prChartOpts:      any = {};
  imbalanceChartOpts: any = {};

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    // Each call wrapped in catchError so one failure doesn't block the whole dashboard
    forkJoin({
      summary:      this.api.getSummary()            .pipe(catchError(() => of({}))),
      channel:      this.api.getFraudByChannel()     .pipe(catchError(() => of([]))),
      country:      this.api.getFraudByCountry()     .pipe(catchError(() => of([]))),
      amount:       this.api.getAmountDistribution() .pipe(catchError(() => of({}))),
      trend:        this.api.getDailyFraudTrend()    .pipe(catchError(() => of({}))),
      score:        this.api.getScoreDistribution()  .pipe(catchError(() => of({}))),
      imbalance:    this.api.getImbalanceReport()    .pipe(catchError(() => of({}))),
      topFlagged:   this.api.getTopFlagged(10)       .pipe(catchError(() => of([]))),
      prCurve:      this.api.getPRCurve()            .pipe(catchError(() => of({}))),
      reviewDepth:  this.api.getReviewQueueDepth()   .pipe(catchError(() => of({}))),
      models:       this.api.getModelComparison()    .pipe(catchError(() => of([]))),
    }).subscribe({
      next: (r: any) => {
        this.summary           = r.summary     || {};
        this.fraudByChannel    = Array.isArray(r.channel)    ? r.channel    : [];
        this.fraudByCountry    = Array.isArray(r.country)    ? r.country    : [];
        this.amountDist        = r.amount      || {};
        this.dailyTrend        = r.trend       || {};
        this.scoreDistribution = r.score       || {};
        this.imbalanceReport   = r.imbalance   || {};
        this.topFlagged        = Array.isArray(r.topFlagged) ? r.topFlagged : [];
        this.prCurve           = r.prCurve     || {};
        this.reviewStats       = r.reviewDepth || {};
        this.modelComparison   = Array.isArray(r.models) ? r.models : [];

        // Hero metric — recall_fraud from champion
        const recall = this.summary?.champion_recall ?? this.summary?.recall_fraud ?? null;
        if (recall !== null && recall !== undefined && !isNaN(+recall)) {
          const pct = Math.round(+recall * 100);
          this.recallPct        = pct + '%';
          this.recallDashOffset = 314 * (1 - +recall);
        }

        this.buildCharts();
        this.loading = false;
      },
      error: (e) => {
        console.error('[Dashboard] forkJoin error:', e);
        this.loading = false;
      },
    });
  }

  private buildCharts(): void {
    // Channel bar chart
    if (this.fraudByChannel.length) {
      const channels = this.fraudByChannel.map((r: any) => r.channel);
      const rates    = this.fraudByChannel.map((r: any) => +(r.fraud_rate * 100).toFixed(2));
      const totals   = this.fraudByChannel.map((r: any) => r.total);
      this.channelChartOpts = {
        tooltip: { trigger: 'axis' },
        legend:  { data: ['Fraud Rate (%)', 'Total Txns'], textStyle: { color: '#94a3b8' } },
        xAxis:   { type: 'category', data: channels, axisLabel: { color: '#64748b', rotate: 15 } },
        yAxis:   [
          { type: 'value', name: 'Fraud %', axisLabel: { color: '#64748b', formatter: '{value}%' } },
          { type: 'value', name: 'Total',   axisLabel: { color: '#64748b' } },
        ],
        series:  [
          { name: 'Fraud Rate (%)', type: 'bar',  data: rates,  itemStyle: { color: '#ef4444' }, yAxisIndex: 0 },
          { name: 'Total Txns',     type: 'line', data: totals, itemStyle: { color: '#3b82f6' }, yAxisIndex: 1, smooth: true },
        ],
        backgroundColor: 'transparent',
      };
    }

    // Country pie
    if (this.fraudByCountry.length) {
      this.countryChartOpts = {
        tooltip: { trigger: 'item', formatter: '{b}: {c} fraud ({d}%)' },
        legend:  { bottom: 0, textStyle: { color: '#94a3b8' } },
        series:  [{
          type: 'pie', radius: ['45%', '70%'], center: ['50%', '45%'],
          data: this.fraudByCountry.map((r: any) => ({ name: r.country, value: r.fraud_count })),
          label: { color: '#94a3b8' },
        }],
        backgroundColor: 'transparent',
      };
    }

    // Amount distribution
    if (this.amountDist?.labels?.length) {
      this.amountChartOpts = {
        tooltip: { trigger: 'axis' },
        legend:  { data: ['Fraud', 'Legit'], textStyle: { color: '#94a3b8' } },
        xAxis:   { type: 'category', data: this.amountDist.labels, axisLabel: { color: '#64748b', rotate: 30, interval: 4 } },
        yAxis:   { type: 'value', axisLabel: { color: '#64748b' } },
        series:  [
          { name: 'Fraud', type: 'bar', data: this.amountDist.fraud, itemStyle: { color: '#ef4444', opacity: .85 }, stack: 'a' },
          { name: 'Legit', type: 'bar', data: this.amountDist.legit, itemStyle: { color: '#3b82f6', opacity: .6 },  stack: 'a' },
        ],
        backgroundColor: 'transparent',
      };
    }

    // Daily fraud trend
    if (this.dailyTrend?.buckets?.length) {
      this.trendChartOpts = {
        tooltip: { trigger: 'axis' },
        legend:  { data: ['Fraud', 'Legit'], textStyle: { color: '#94a3b8' } },
        xAxis:   { type: 'category', data: this.dailyTrend.buckets, axisLabel: { color: '#64748b', interval: 4 } },
        yAxis:   { type: 'value', axisLabel: { color: '#64748b' } },
        series:  [
          { name: 'Fraud', type: 'line', data: this.dailyTrend.fraud, smooth: true, areaStyle: { opacity: .25 }, itemStyle: { color: '#ef4444' } },
          { name: 'Legit', type: 'line', data: this.dailyTrend.legit, smooth: true, areaStyle: { opacity: .15 }, itemStyle: { color: '#10b981' } },
        ],
        backgroundColor: 'transparent',
      };
    }

    // Score distribution
    if (this.scoreDistribution?.bins?.length) {
      this.scoreChartOpts = {
        tooltip: { trigger: 'axis' },
        xAxis:   { type: 'category', data: this.scoreDistribution.bins, axisLabel: { color: '#64748b', rotate: 30, interval: 3 } },
        yAxis:   { type: 'value', axisLabel: { color: '#64748b' } },
        series:  [{
          type: 'bar', data: this.scoreDistribution.counts,
          itemStyle: { color: (p: any) => {
            const idx = p.dataIndex / this.scoreDistribution.bins.length;
            return idx > .7 ? '#ef4444' : idx > .3 ? '#f59e0b' : '#10b981';
          }},
        }],
        backgroundColor: 'transparent',
      };
    }

    // PR Curve
    if (this.prCurve?.precision?.length) {
      this.prChartOpts = {
        tooltip: { trigger: 'axis' },
        xAxis:   { type: 'value', name: 'Recall',    min: 0, max: 1, axisLabel: { color: '#64748b' } },
        yAxis:   { type: 'value', name: 'Precision', min: 0, max: 1, axisLabel: { color: '#64748b' } },
        series:  [{
          type: 'line', smooth: true, showSymbol: false,
          data: this.prCurve.recall.map((r: number, i: number) => [r, this.prCurve.precision[i]]),
          itemStyle: { color: '#3b82f6' }, areaStyle: { opacity: .2 },
        }],
        backgroundColor: 'transparent',
      };
    }

    // Imbalance pie
    if (this.imbalanceReport?.fraud_count !== undefined && !this.imbalanceReport?.error) {
      this.imbalanceChartOpts = {
        tooltip: { trigger: 'item' },
        series: [{
          type: 'pie', radius: ['50%', '72%'], center: ['50%', '50%'],
          data: [
            { name: 'Fraud', value: this.imbalanceReport.fraud_count, itemStyle: { color: '#ef4444' } },
            { name: 'Legit', value: this.imbalanceReport.legit_count, itemStyle: { color: '#10b981' } },
          ],
          label: { color: '#94a3b8', formatter: '{b}: {d}%' },
        }],
        backgroundColor: 'transparent',
      };
    }
  }

  riskClass(prob: number): string {
    if (prob >= 0.7) return 'risk--high';
    if (prob >= 0.3) return 'risk--medium';
    return 'risk--low';
  }

  metricClass(val: number): string {
    if (val >= 0.85) return 'mc--excellent';
    if (val >= 0.70) return 'mc--good';
    if (val >= 0.55) return 'mc--fair';
    return 'mc--poor';
  }

  compositeScore(m: any): number {
    return +(
      0.50 * (m.recall_fraud  || 0) +
      0.20 * (m.pr_auc        || 0) +
      0.20 * (m.auc_roc       || 0) +
      0.10 * (m.f1_fraud      || 0)
    ).toFixed(4);
  }
}
