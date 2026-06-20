import { APP_INITIALIZER, ApplicationConfig, provideBrowserGlobalErrorListeners, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { provideEchartsCore } from 'ngx-echarts';
import * as echarts from 'echarts/core';
import {
  BarChart, LineChart, PieChart, ScatterChart, RadarChart,
  HeatmapChart, BoxplotChart, CustomChart,
} from 'echarts/charts';
import {
  GridComponent, TooltipComponent, LegendComponent,
  TitleComponent, DataZoomComponent, VisualMapComponent,
  PolarComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

import { routes } from './app.routes';
import { authInterceptor } from './core/auth.interceptor';
import { AuthService } from './core/auth.service';
import { ThemeService } from './core/theme.service';

echarts.use([
  BarChart, LineChart, PieChart, ScatterChart, RadarChart,
  HeatmapChart, BoxplotChart, CustomChart,
  GridComponent, TooltipComponent, LegendComponent,
  TitleComponent, DataZoomComponent, VisualMapComponent,
  PolarComponent,
  CanvasRenderer,
]);

function initAuth(auth: AuthService) {
  return () => auth.checkSession().toPromise();
}

// Eagerly create ThemeService so its effect() fires immediately and
// applies the correct data-theme attribute before first render.
function initTheme(theme: ThemeService) {
  return () => { /* ThemeService constructor + effect() does the work */ };
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideBrowserGlobalErrorListeners(),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideHttpClient(withInterceptors([authInterceptor])),
    provideEchartsCore({ echarts }),
    {
      provide:    APP_INITIALIZER,
      useFactory: initTheme,
      deps:       [ThemeService],
      multi:      true,
    },
    {
      provide:    APP_INITIALIZER,
      useFactory: initAuth,
      deps:       [AuthService],
      multi:      true,
    },
  ],
};
