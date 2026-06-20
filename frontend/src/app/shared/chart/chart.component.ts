import { Component, Input, OnChanges, SimpleChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgxEchartsDirective } from 'ngx-echarts';
import type { EChartsOption } from 'echarts';

@Component({
  selector: 'app-chart',
  standalone: true,
  imports: [CommonModule, NgxEchartsDirective],
  template: `
    <div class="chart-wrap" [style.height]="height">
      <div echarts [options]="options" [theme]="'default'"
           class="chart-inner" [loading]="loading"></div>
    </div>
  `,
  styles: [`
    .chart-wrap { width: 100%; }
    .chart-inner { width: 100%; height: 100%; }
  `],
})
export class ChartComponent implements OnChanges {
  @Input() options: EChartsOption = {};
  @Input() height = '280px';
  @Input() loading = false;

  ngOnChanges(_: SimpleChanges): void {}
}
