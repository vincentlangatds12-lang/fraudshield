import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/api.service';

@Component({
  selector: 'app-transactions',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './transactions.component.html',
  styleUrls: ['./transactions.component.scss'],
})
export class TransactionsComponent implements OnInit {
  loading   = false;
  items:    any[] = [];
  total     = 0;
  page      = 1;
  pageSize  = 25;

  // Filters
  filterSplit   = '';
  filterChannel = '';
  filterCountry = '';
  filterFraud   = '';
  filterMinProb = '';

  channels  = ['', 'mobile_money', 'p2p', 'bank_transfer', 'card', 'airtime', 'bill_pay'];
  countries = ['', 'KE', 'NG'];

  constructor(private api: ApiService) {}

  ngOnInit(): void { this.load(); }

  load(): void {
    this.loading = true;
    const filters: any = {};
    if (this.filterSplit)   filters['split']    = this.filterSplit;
    if (this.filterChannel) filters['channel']  = this.filterChannel;
    if (this.filterCountry) filters['country']  = this.filterCountry;
    if (this.filterFraud !== '') filters['is_fraud'] = parseInt(this.filterFraud);
    if (this.filterMinProb) filters['min_prob'] = parseFloat(this.filterMinProb);

    this.api.getTransactions(this.page, this.pageSize, filters).subscribe({
      next: (r) => { this.items = r.transactions || []; this.total = r.total || 0; this.loading = false; },
      error: () => { this.loading = false; },
    });
  }

  applyFilters(): void { this.page = 1; this.load(); }
  clearFilters(): void {
    this.filterSplit = ''; this.filterChannel = ''; this.filterCountry = '';
    this.filterFraud = ''; this.filterMinProb = '';
    this.page = 1; this.load();
  }

  nextPage(): void  { if (this.page < this.totalPages) { this.page++; this.load(); } }
  prevPage(): void  { if (this.page > 1) { this.page--; this.load(); } }
  goPage(p: number) { this.page = p; this.load(); }

  get totalPages(): number { return Math.ceil(this.total / this.pageSize) || 1; }
  get pageRange(): number[] {
    const p = this.totalPages;
    const c = this.page;
    const start = Math.max(1, c - 2);
    const end   = Math.min(p, c + 2);
    return Array.from({ length: end - start + 1 }, (_, i) => start + i);
  }

  riskClass(p: number | null): string {
    if (p === null || p === undefined) return '';
    if (p >= 0.7) return 'risk--high';
    if (p >= 0.3) return 'risk--medium';
    return 'risk--low';
  }
}
