import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule }                from '@angular/common';
import { FormsModule }                 from '@angular/forms';
import { HttpClient }                  from '@angular/common/http';
import { interval, Subscription }      from 'rxjs';
import { switchMap, takeWhile }        from 'rxjs/operators';
import { ApiService }                  from '../../core/api.service';

type Tab = 'pipeline' | 'mlflow' | 'versions';

interface PipelineStep {
  id:     number;
  label:  string;
  detail: string;
  status: 'pending' | 'running' | 'done' | 'error';
}

@Component({
  selector:    'app-training',
  standalone:  true,
  imports:     [CommonModule, FormsModule],
  templateUrl: './training.component.html',
  styleUrls:   ['./training.component.scss'],
})
export class TrainingComponent implements OnInit, OnDestroy {

  activeTab: Tab = 'pipeline';

  // ── Ingest state ────────────────────────────────────────────────────────
  ingestRunning   = false;
  ingestDone      = false;
  ingestError     = '';
  dbCount         = 0;
  ingestPoll?: Subscription;

  // ── Pipeline state ───────────────────────────────────────────────────────
  pipelineRunning   = false;
  pipelineDone      = false;
  pipelineError     = '';
  lastRunId         = '';
  flamlBudget       = 120;
  pipelinePoll?: Subscription;

  steps: PipelineStep[] = [
    { id: 1,  label: 'Load raw data',                    detail: 'train.csv · test.csv · identity.csv',                                      status: 'pending' },
    { id: 2,  label: 'Data integrity checks',             detail: 'Temporal split · leakage guard · currency scale',                          status: 'pending' },
    { id: 3,  label: 'Feature engineering (20+ groups)',  detail: 'Velocity · target enc · frequency · network · V*/C*/D* · percentile rank', status: 'pending' },
    { id: 4,  label: 'Imbalance analysis',                detail: 'class_weight · SMOTE · ADASYN — picks best by PR-AUC',                     status: 'pending' },
    { id: 5,  label: 'Temporal train/val split',          detail: 'Last 20% by TransactionDT — simulates real deployment',                    status: 'pending' },
    { id: 6,  label: 'Train Logistic Regression',         detail: 'sklearn · balanced class weight · saved to DB immediately',                status: 'pending' },
    { id: 7,  label: 'FLAML → LightGBM',                  detail: '300s budget · metric=recall · saved to DB immediately',                   status: 'pending' },
    { id: 8,  label: 'FLAML → CatBoost',                  detail: '300s budget · metric=recall · saved to DB immediately',                   status: 'pending' },
    { id: 9,  label: 'FLAML → Random Forest',             detail: '300s budget · metric=recall · saved to DB immediately',                   status: 'pending' },
    { id: 10, label: 'FLAML → XGBoost',                   detail: '300s budget · metric=recall · saved to DB immediately',                   status: 'pending' },
    { id: 11, label: 'Log all runs to MLflow',             detail: 'Metrics · hyperparams · artifacts for all 5 models',                     status: 'pending' },
    { id: 12, label: 'SHAP global explanations',           detail: 'Mean |SHAP| per feature for champion model',                              status: 'pending' },
    { id: 13, label: 'Generate predictions.csv',           detail: 'Champion model scores all test transactions — submission ready',          status: 'pending' },
  ];

  // ── MLflow runs ──────────────────────────────────────────────────────────
  mlflowRuns:    any[] = [];
  mlflowUrl      = '';
  mlflowLoading  = false;

  // ── Model versions ────────────────────────────────────────────────────────
  modelVersions:    any[] = [];
  versionsLoading   = false;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.refreshIngestStatus();
    this.refreshPipelineStatus();
    this.loadMLflowUrl();
  }

  ngOnDestroy(): void {
    this.ingestPoll?.unsubscribe();
    this.pipelinePoll?.unsubscribe();
  }

  // ── Tab switching ─────────────────────────────────────────────────────────
  setTab(tab: Tab): void {
    this.activeTab = tab;
    if (tab === 'mlflow')   this.loadMlflowRuns();
    if (tab === 'versions') this.loadModelVersions();
  }

  // ── Ingest ────────────────────────────────────────────────────────────────
  ingestData(): void {
    this.ingestError  = '';
    this.ingestDone   = false;
    this.ingestRunning = true;
    this.api.ingestData().subscribe({
      next: () => this.pollIngest(),
      error: e  => { this.ingestError = e?.error?.detail || 'Ingest failed'; this.ingestRunning = false; },
    });
  }

  private pollIngest(): void {
    this.ingestPoll?.unsubscribe();
    this.ingestPoll = interval(2000).pipe(
      switchMap(() => this.api.getIngestStatus()),
    ).subscribe({
      next: s => {
        this.dbCount = s.db_transaction_count ?? 0;
        if (!s.running) {
          this.ingestRunning = false;
          this.ingestPoll?.unsubscribe();
          if (s.error) { this.ingestError = s.error; }
          else          { this.ingestDone = true; }
        }
      },
      error: () => { this.ingestRunning = false; this.ingestPoll?.unsubscribe(); },
    });
  }

  private refreshIngestStatus(): void {
    this.api.getIngestStatus().subscribe({
      next: s => {
        this.dbCount       = s.db_transaction_count ?? 0;
        this.ingestRunning = s.running;
        this.ingestDone    = this.dbCount > 0 && !s.running;
        if (s.running) this.pollIngest();
      },
    });
  }

  // ── Pipeline ─────────────────────────────────────────────────────────────
  runPipeline(): void {
    this.pipelineError = '';
    this.pipelineDone  = false;
    this.resetSteps();
    this.pipelineRunning = true;
    this.animateSteps();
    this.api.triggerPipeline({ flaml_budget: this.flamlBudget }).subscribe({
      next: () => this.pollPipeline(),
      error: e  => {
        this.pipelineError   = e?.error?.detail || 'Failed to start pipeline';
        this.pipelineRunning = false;
        this.resetSteps();
      },
    });
  }

  private pollPipeline(): void {
    this.pipelinePoll?.unsubscribe();
    this.pipelinePoll = interval(3000).pipe(
      switchMap(() => this.api.getPipelineStatus()),
    ).subscribe({
      next: s => {
        if (!s.running) {
          this.pipelineRunning = false;
          this.pipelinePoll?.unsubscribe();
          if (s.last_error) {
            this.pipelineError = s.last_error;
            this.markStepError();
          } else {
            this.lastRunId    = s.last_run_id || '';
            this.pipelineDone = true;
            this.markAllDone();
            // Auto re-evaluate champion with correct weights
            this.api.rechampion().subscribe();
          }
        }
      },
      error: () => { this.pipelineRunning = false; this.pipelinePoll?.unsubscribe(); },
    });
  }

  private refreshPipelineStatus(): void {
    this.api.getPipelineStatus().subscribe({
      next: s => {
        this.pipelineRunning = s.running;
        this.lastRunId       = s.last_run_id || '';
        if (s.running) { this.animateSteps(); this.pollPipeline(); }
        else if (this.lastRunId) { this.markAllDone(); this.pipelineDone = true; }
      },
    });
  }

  private _stepTimer?: any;
  private _stepIndex  = 0;

  private animateSteps(): void {
    this.resetSteps();
    this._stepIndex = 0;
    clearInterval(this._stepTimer);
    // 13 steps, ~5min per FLAML model = ~25min total, so ~115s per step
    this._stepTimer = setInterval(() => {
      if (this._stepIndex < this.steps.length) {
        if (this._stepIndex > 0) this.steps[this._stepIndex - 1].status = 'done';
        this.steps[this._stepIndex].status = 'running';
        this._stepIndex++;
      } else {
        clearInterval(this._stepTimer);
      }
    }, 5000);
  }

  private resetSteps(): void {
    clearInterval(this._stepTimer);
    this.steps.forEach(s => s.status = 'pending');
    this._stepIndex = 0;
  }

  private markAllDone(): void {
    clearInterval(this._stepTimer);
    this.steps.forEach(s => s.status = 'done');
  }

  private markStepError(): void {
    clearInterval(this._stepTimer);
    const running = this.steps.find(s => s.status === 'running');
    if (running) running.status = 'error';
  }

  // ── MLflow ────────────────────────────────────────────────────────────────
  loadMlflowRuns(): void {
    this.mlflowLoading = true;
    this.api.getMlflowRuns().subscribe({
      next: runs  => { this.mlflowRuns = runs || []; this.mlflowLoading = false; },
      error: ()   => { this.mlflowLoading = false; },
    });
  }

  rechampionLoading = false;
  rechampionResult: any = null;

  rechampion(): void {
    this.rechampionLoading = true;
    this.api.rechampion().subscribe({
      next: (r) => {
        this.rechampionResult  = r;
        this.rechampionLoading = false;
        // Refresh versions list
        if (this.activeTab === 'versions') this.loadModelVersions();
      },
      error: () => { this.rechampionLoading = false; },
    });
  }

  private loadMLflowUrl(): void {
    this.api.getMlflowUrl().subscribe({ next: r => this.mlflowUrl = r?.url || '' });
  }

  // ── Model versions ────────────────────────────────────────────────────────
  loadModelVersions(): void {
    this.versionsLoading = true;
    this.api.getModelVersions().subscribe({
      next: v   => { this.modelVersions = v || []; this.versionsLoading = false; },
      error: () => { this.versionsLoading = false; },
    });
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  metricColor(val: number): string {
    if (val >= 0.85) return 'metric--excellent';
    if (val >= 0.70) return 'metric--good';
    if (val >= 0.55) return 'metric--fair';
    return 'metric--poor';
  }

  formatDuration(s: number): string {
    if (s < 60)   return `${s.toFixed(0)}s`;
    if (s < 3600) return `${(s / 60).toFixed(1)}m`;
    return `${(s / 3600).toFixed(1)}h`;
  }
}
