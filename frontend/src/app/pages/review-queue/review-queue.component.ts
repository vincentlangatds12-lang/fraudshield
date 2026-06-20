import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';
import { ChartComponent } from '../../shared/chart/chart.component';
import { StatCardComponent } from '../../shared/stat-card/stat-card.component';

@Component({
  selector: 'app-review-queue',
  standalone: true,
  imports: [CommonModule, FormsModule, ChartComponent, StatCardComponent],
  templateUrl: './review-queue.component.html',
  styleUrls: ['./review-queue.component.scss'],
})
export class ReviewQueueComponent implements OnInit {
  loading       = true;
  activeTab: 'pending' | 'reviewed' = 'pending';
  stats:    any = {};
  items:    any[] = [];
  reviewed: any[] = [];
  page      = 1;
  total     = 0;
  totalReviewed = 0;

  // Decision modal
  showModal   = false;
  selectedItem: any = null;
  decision    = 'confirmed_fraud';
  analystNote = '';
  submitting  = false;
  submitMsg   = '';

  // Charts
  decisionsChartOpts: any = {};
  scoreHistOpts:      any = {};
  populatingQueue     = false;

  constructor(private api: ApiService) {}

  ngOnInit(): void { this.loadAll(); }

  loadAll(): void {
    this.loading = true;
    Promise.all([
      this.api.getReviewStats().toPromise(),
      this.api.getReviewQueue(1, 'pending').toPromise(),
      this.api.getReviewQueue(1, 'reviewed').toPromise(),
      this.api.getReviewDecisions().toPromise(),
    ]).then(([stats, pending, rev, decisions]) => {
      this.stats    = stats || {};
      this.items    = pending?.items   || [];
      this.total    = pending?.total   || 0;
      this.reviewed = rev?.items       || [];
      this.totalReviewed = rev?.total  || 0;
      this.buildDecisionChart(decisions || {});
      this.buildScoreChart();
      this.loading = false;
    }).catch(() => { this.loading = false; });
  }

  openDecision(item: any): void {
    this.selectedItem = item;
    this.decision     = item.fraud_prob >= 0.5 ? 'confirmed_fraud' : 'confirmed_legit';
    this.analystNote  = '';
    this.submitMsg    = '';
    this.showModal    = true;
  }

  submitDecision(): void {
    if (!this.selectedItem) return;
    this.submitting = true;
    this.api.submitDecision({
      queue_item_id: this.selectedItem.queue_item_id,
      decision:      this.decision,
      analyst_label: this.decision === 'confirmed_fraud' ? 1 : 0,
      notes:         this.analystNote,
    }).subscribe({
      next: () => {
        this.submitting = false;
        this.showModal  = false;
        this.submitMsg  = 'Decision saved.';
        this.loadAll();
      },
      error: (e) => {
        this.submitting = false;
        this.submitMsg  = e?.error?.detail || 'Error saving decision';
      },
    });
  }

  populateQueue(): void {
    this.populatingQueue = true;
    this.api.populateReviewQueue().subscribe({
      next: (r) => {
        this.populatingQueue = false;
        this.loadAll();
      },
      error: () => { this.populatingQueue = false; },
    });
  }

  private buildDecisionChart(decisions: any): void {
    const labels = Object.keys(decisions);
    const values = Object.values(decisions);
    if (!labels.length) return;
    const colours: Record<string, string> = {
      confirmed_fraud: '#ef4444', confirmed_legit: '#10b981', uncertain: '#f59e0b',
    };
    this.decisionsChartOpts = {
      tooltip: { trigger: 'item' },
      legend:  { bottom: 0, textStyle: { color: '#94a3b8' } },
      series: [{
        type: 'pie', radius: ['45%', '70%'],
        data: labels.map((l, i) => ({
          name:  l.replace(/_/g, ' '), value: values[i],
          itemStyle: { color: colours[l] || '#94a3b8' },
        })),
        label: { color: '#94a3b8', formatter: '{b}: {d}%' },
      }],
      backgroundColor: 'transparent',
    };
  }

  private buildScoreChart(): void {
    if (!this.items.length && !this.reviewed.length) return;
    const all  = [...this.items, ...this.reviewed];
    const bins = Array.from({ length: 10 }, (_, i) => i / 10);
    const counts = bins.map(b => all.filter((x: any) => x.fraud_prob >= b && x.fraud_prob < b + 0.1).length);
    this.scoreHistOpts = {
      tooltip: { trigger: 'axis' },
      xAxis:   { type: 'category', data: bins.map(b => `${(b*100).toFixed(0)}-${((b+.1)*100).toFixed(0)}%`), axisLabel: { color: '#64748b' } },
      yAxis:   { type: 'value', axisLabel: { color: '#64748b' } },
      series:  [{ type: 'bar', data: counts, itemStyle: { color: (p: any) => p.dataIndex >= 7 ? '#ef4444' : p.dataIndex >= 3 ? '#f59e0b' : '#10b981' } }],
      backgroundColor: 'transparent',
    };
  }

  riskClass(p: number): string { return p >= 0.7 ? 'risk--high' : p >= 0.3 ? 'risk--medium' : 'risk--low'; }
}
