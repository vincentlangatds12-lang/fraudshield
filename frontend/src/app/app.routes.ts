import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: 'login', loadComponent: () => import('./pages/login/login.component').then(m => m.LoginComponent) },

  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },

  { path: 'dashboard',        loadComponent: () => import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent) },
  { path: 'transactions',     loadComponent: () => import('./pages/transactions/transactions.component').then(m => m.TransactionsComponent) },
  { path: 'review-queue',     loadComponent: () => import('./pages/review-queue/review-queue.component').then(m => m.ReviewQueueComponent) },
  { path: 'model-comparison', loadComponent: () => import('./pages/model-comparison/model-comparison.component').then(m => m.ModelComparisonComponent) },
  { path: 'model-monitoring', loadComponent: () => import('./pages/model-monitoring/model-monitoring.component').then(m => m.ModelMonitoringComponent) },
  { path: 'explainability',   loadComponent: () => import('./pages/explainability/explainability.component').then(m => m.ExplainabilityComponent) },
  { path: 'training',         loadComponent: () => import('./pages/training/training.component').then(m => m.TrainingComponent) },
  { path: 'analytics-3d',     loadComponent: () => import('./pages/analytics-3d/analytics-3d.component').then(m => m.Analytics3dComponent) },

  { path: '**', redirectTo: 'dashboard' },
];
