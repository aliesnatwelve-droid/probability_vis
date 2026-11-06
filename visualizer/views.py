"""
Views for the probability visualizer application.
"""
import json
import io
import base64
import csv
import re

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt

from . import stats_engine as se

# ---------------------------------------------------------------------------
# Utility: build a Plotly figure from distribution data
# ---------------------------------------------------------------------------
COLORS = {
    'primary': '#40E0D0',
    'secondary': '#001F3F',
    'accent1': '#FF6F61',
    'accent2': '#6B5B95',
    'accent3': '#88B04B',
    'bg': '#F8F9FA',
    'text': '#001F3F',
}


def _make_plotly_layout(title='', xlabel='x', ylabel='Density', height=450):
    """Common Plotly layout with the Turquoise/Navy theme."""
    return go.Layout(
        title=title,
        xaxis=dict(title=xlabel, showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='#ddd'),
        yaxis=dict(title=ylabel, showgrid=True, gridcolor='#eee', zeroline=True, zerolinecolor='#ddd'),
        template='plotly_white',
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode='x unified',
        font=dict(family='Segoe UI, Arial, sans-serif', color=COLORS['text']),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
    )


def _fig_to_json(fig):
    """Convert a Plotly figure to a JSON-serializable spec dict."""
    return json.loads(fig.to_json())


def generate_distribution_plots(slug, **params):
    """
    Generate PDF and CDF plots for a given distribution.
    Returns JSON figure specs for embedding.
    """
    # Determine x-range based on distribution
    if slug == 'uniform':
        a = float(params.get('a', 0))
        b = float(params.get('b', 1))
        x_pdf = np.linspace(a - 0.5, b + 0.5, 500)
        x_cdf = np.linspace(a - 0.5, b + 0.5, 500)
    elif slug == 'exponential':
        rate = float(params.get('rate', 1))
        x_pdf = np.linspace(0, 8 / rate, 500)
        x_cdf = np.linspace(0, 8 / rate, 500)
    elif slug in ('poisson', 'binomial', 'bernoulli', 'geometric', 'negative_binomial'):
        # Discrete — use integer domain
        if slug == 'poisson':
            lam = float(params.get('lam', 3))
            xs = np.arange(0, max(lam * 3, 15), 1)
        elif slug == 'binomial':
            n = int(params.get('n', 10))
            xs = np.arange(0, int(n) + 1)
        elif slug == 'bernoulli':
            xs = np.array([0, 1])
        elif slug == 'geometric':
            p = float(params.get('p', 0.3))
            xs = np.arange(1, min(int(10 / p), 100))
        elif slug == 'negative_binomial':
            r = int(params.get('r', 5))
            p = float(params.get('p', 0.5))
            xs = np.arange(0, min(int(r * 4 / p), 200))
        else:
            xs = np.arange(0, 25)
        dist, args = se._build_params(slug, **params)
        pmf_vals = dist.pmf(xs, **args)
        cdf_vals = dist.cdf(xs, **args)

        fig_pmf = go.Figure(
            data=[go.Bar(x=xs, y=pmf_vals, marker_color=COLORS['primary'], opacity=0.85,
                         hovertemplate='x=%{x}<br>P(X=x)=%{y:.4f}<extra></extra>')],
            layout=_make_plotly_layout('Probability Mass Function', xlabel='x', ylabel='P(X = x)')
        )
        fig_cdf = go.Figure(
            data=[go.Scatter(x=xs, y=cdf_vals, mode='lines+markers',
                             marker_color=COLORS['secondary'], line_shape='hv',
                             hovertemplate='x=%{x}<br>F(x)=%{y:.4f}<extra></extra>')],
            layout=_make_plotly_layout('Cumulative Distribution Function', xlabel='x', ylabel='F(x)')
        )
        return _fig_to_json(fig_pmf), _fig_to_json(fig_cdf)

    else:
        # Continuous — auto-range from distribution properties
        dist, args = se._build_params(slug, **params)
        try:
            lo = float(dist.ppf(0.001, **args))
            hi = float(dist.ppf(0.999, **args))
        except Exception:
            lo, hi = -5, 5
        if not np.isfinite(lo):
            lo = -5
        if not np.isfinite(hi):
            hi = 5
        padding = (hi - lo) * 0.1
        x_pdf = np.linspace(lo - padding, hi + padding, 800)
        x_cdf = np.linspace(lo - padding, hi + padding, 800)

    # Generate plots for continuous case
    x_p, y_p = se.pdf(slug, x_pdf, **params)
    x_c, y_c = se.cdf(slug, x_cdf, **params)

    # PDF plot
    fig_pdf = go.Figure(
        data=[go.Scatter(x=x_p, y=y_p, mode='lines', fill='tozeroy',
                         line=dict(color=COLORS['primary'], width=3),
                         hovertemplate='x=%{x:.3f}<br>f(x)=%{y:.4f}<extra></extra>')],
        layout=_make_plotly_layout('Probability Density Function', xlabel='x', ylabel='f(x)')
    )

    # Add mean line if normal
    if slug == 'normal':
        mu = float(params.get('mu', 0))
        fig_pdf.add_vline(x=mu, line_dash='dash', line_color=COLORS['accent1'],
                          annotation_text=f'μ={mu}', annotation_font_color=COLORS['accent1'])

    # CDF plot
    fig_cdf = go.Figure(
        data=[go.Scatter(x=x_c, y=y_c, mode='lines',
                         line=dict(color=COLORS['secondary'], width=3),
                         fill='tozeroy', fillcolor='rgba(0,31,63,0.08)',
                         hovertemplate='x=%{x:.3f}<br>F(x)=%{y:.4f}<extra></extra>')],
        layout=_make_plotly_layout('Cumulative Distribution Function', xlabel='x', ylabel='F(x)')
    )

    return _fig_to_json(fig_pdf), _fig_to_json(fig_cdf)


def generate_side_by_side(slug1, slug2, params1, params2):
    """Generate side-by-side comparison of two distributions."""
    # Determine range
    try:
        d1, a1 = se._build_params(slug1, **params1)
        d2, a2 = se._build_params(slug2, **params2)
        lo1, hi1 = float(d1.ppf(0.001, **a1)), float(d1.ppf(0.999, **a1))
        lo2, hi2 = float(d2.ppf(0.001, **a2)), float(d2.ppf(0.999, **a2))
        lo = min(lo1, lo2)
        hi = max(hi1, hi2)
    except Exception:
        lo, hi = -5, 5

    x = np.linspace(lo, hi, 500)
    y1, _ = se.pdf(slug1, x, **params1)
    y2, _ = se.pdf(slug2, x, **params2)

    name1 = se.DISTRIBUTIONS.get(slug1, {}).get('name', slug1)
    name2 = se.DISTRIBUTIONS.get(slug2, {}).get('name', slug2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=name1,
                              line=dict(color=COLORS['primary'], width=3)))
    fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=name2,
                              line=dict(color=COLORS['accent1'], width=3)))
    fig.update_layout(
        title=dict(text=f'{name1} vs {name2}', x=0.5),
        xaxis=dict(title='x', showgrid=True, gridcolor='#eee'),
        yaxis=dict(title='Density', showgrid=True, gridcolor='#eee'),
        template='plotly_white',
        height=450,
        hovermode='x unified',
        font=dict(family='Segoe UI, Arial, sans-serif', color=COLORS['text']),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    return _fig_to_json(fig)


def generate_joint_plot(slug, n_samples=1000, **params):
    """Generate a joint distribution summary (histogram + density) from samples."""
    samples = se.sample(slug, n_samples=n_samples, **params)
    name = se.DISTRIBUTIONS.get(slug, {}).get('name', slug)

    fig = make_subplots(rows=2, cols=2, column_widths=[0.8, 0.2], row_heights=[0.2, 0.8],
                        specs=[[{'type': 'xy'}, None], [{'type': 'xy'}, None]])

    # Main histogram
    fig.add_trace(
        go.Histogram(x=samples, nbinsx=50, marker_color=COLORS['primary'],
                     opacity=0.75, name='Samples', showlegend=False),
        row=2, col=1
    )

    # Density overlay
    x_density = np.linspace(np.min(samples), np.max(samples), 200)
    dist, args = se._build_params(slug, **params)
    y_pdf = dist.pdf(x_density, **args) if hasattr(dist, 'pdf') else dist.pmf(x_density, **args)
    fig.add_trace(
        go.Scatter(x=x_density, y=y_pdf, mode='lines',
                   line=dict(color=COLORS['accent1'], width=2),
                   name='PDF', showlegend=True),
        row=2, col=1
    )

    # Rug plot on top
    rug_samples = samples[:200]
    fig.add_trace(
        go.Scatter(x=rug_samples, y=[0] * len(rug_samples), mode='markers',
                   marker=dict(color=COLORS['secondary'], size=3, symbol='line-ns-open'),
                   name='Rug', showlegend=True),
        row=1, col=1
    )

    fig.update_layout(
        title=dict(text=f'Joint Distribution — {name} ({n_samples} samples)', x=0.5),
        template='plotly_white',
        height=500,
        hovermode='x unified',
        font=dict(family='Segoe UI, Arial, sans-serif', color=COLORS['text']),
        plot_bgcolor='rgba(255,255,255,0)',
        paper_bgcolor='rgba(255,255,255,0)',
        bargap=0.05,
    )
    fig.update_xaxes(title_text='x', row=2, col=1)
    fig.update_yaxes(title_text='Count', row=2, col=1)

    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# 3D Plotting helpers — f(x, y) over a 2D domain
# ---------------------------------------------------------------------------

_3D_2D_FUNC_NAMES = {
    'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan',
    'sinh', 'cosh', 'tanh',
    'exp', 'log', 'log10', 'log2',
    'sqrt', 'abs', 'power', 'floor', 'ceil',
    'erf', 'erfc', 'gamma',
}


def _eval_2d_function(expr, x_grid, y_grid):
    """
    Safely evaluate a user expression f(x, y) on a 2D meshgrid.
    Uses numexpr for speed and safety; falls back to numpy eval if needed.
    """
    import math
    safe_dict = {
        'x': x_grid, 'y': y_grid,
        'np': np, 'math': math,
        'pi': math.pi, 'e': math.e,
    }
    for name in _3D_2D_FUNC_NAMES:
        safe_dict[name] = getattr(np, name, None)
    # A couple are not in np namespace
    try:
        from scipy.special import erf, erfc, gamma as _gamma
        safe_dict['erf'] = erf
        safe_dict['erfc'] = erfc
        safe_dict['gamma'] = _gamma
    except ImportError:
        pass

    expr = expr.strip()

    # Prefer numexpr (faster + safe C-level eval)
    try:
        import numexpr as ne
        z = ne.evaluate(expr, local_dict=safe_dict)
        return np.asarray(z, dtype=float)
    except Exception:
        pass

    # Fall back to plain Python eval with restricted builtins
    code = compile(expr, '<string>', 'eval')
    z = eval(code, {"__builtins__": {}}, safe_dict)
    return np.asarray(z, dtype=float)


def _build_3d_mask(x_grid, y_grid, condition):
    """
    Build a boolean mask for a 2D region. Returns (mask, error_message).
    `condition` may be empty / None (returns all-True).
    """
    if not condition or not condition.strip():
        return np.ones_like(x_grid, dtype=bool), None

    cond = condition.strip()
    # Common shorthand: x2 → x**2, y2 → y**2
    cond = re.sub(r'([xy])(\d+)', r'\1**\2', cond)
    cond = cond.replace('^', '**')

    try:
        import numexpr as ne
        m = ne.evaluate(cond, local_dict={'x': x_grid, 'y': y_grid})
        return np.asarray(m, dtype=bool), None
    except Exception as ne_err:
        try:
            code = compile(cond, '<string>', 'eval')
            m = eval(code, {"__builtins__": {}},
                     {'x': x_grid, 'y': y_grid, 'np': np,
                      'abs': np.abs, 'sqrt': np.sqrt, 'sin': np.sin,
                      'cos': np.cos, 'exp': np.exp, 'log': np.log})
            return np.asarray(m, dtype=bool), None
        except Exception as e:
            return None, f"Condition error (numexpr: {ne_err}; eval: {e})"


def generate_3d_surface(expr, x_min, x_max, y_min, y_max, resolution,
                       condition='', colorscale='Viridis', opacity=0.9,
                       show_contour=True):
    """Generate a 3D surface plot of f(x, y)."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _eval_2d_function(expr, X, Y)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)

    # Apply region mask (outside region is transparent in surface)
    if condition and condition.strip():
        mask, _ = _build_3d_mask(X, Y, condition)
        if mask is not None:
            Z_masked = np.where(mask, Z, np.nan)
        else:
            Z_masked = Z
    else:
        Z_masked = Z

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z_masked,
            colorscale=colorscale.lower(),
            opacity=opacity,
            colorbar=dict(title='f(x,y)', x=1.02, len=0.7),
            contours={
                'z': dict(show=show_contour, usecolormap=True,
                          highlightcolor='limegreen', project=dict(z=True))
            },
            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>f(x,y)=%{z:.4f}<extra></extra>',
        )
    ])
    fig.update_layout(
        title=dict(text=f'3D Surface: f(x,y) = {expr}', x=0.5,
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='<b>x</b>',
            yaxis_title='<b>y</b>',
            zaxis_title='<b>f(x,y)</b>',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='cube',
        ),
        height=650,
        margin=dict(l=0, r=0, t=60, b=0),
        template='plotly_white',
    )
    return _fig_to_json(fig)


def generate_3d_contour(expr, x_min, x_max, y_min, y_max, resolution,
                        condition='', colorscale='Viridis'):
    """Generate a filled 2D contour plot of f(x, y)."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _eval_2d_function(expr, X, Y)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)

    if condition and condition.strip():
        mask, _ = _build_3d_mask(X, Y, condition)
        Z_show = np.where(mask, Z, np.nan) if mask is not None else Z
    else:
        Z_show = Z

    fig = go.Figure(data=[
        go.Contour(
            x=x, y=y, z=Z_show,
            colorscale=colorscale.lower(),
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(title='f(x,y)', x=1.02),
            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>f(x,y)=%{z:.4f}<extra></extra>',
        )
    ])
    fig.update_layout(
        title=dict(text=f'Contour: f(x,y) = {expr}', x=0.5),
        xaxis_title='x', yaxis_title='y',
        height=600, template='plotly_white',
    )
    return _fig_to_json(fig)


def generate_3d_heatmap(expr, x_min, x_max, y_min, y_max, resolution,
                        condition='', colorscale='Viridis'):
    """Generate a heatmap of f(x, y)."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _eval_2d_function(expr, X, Y)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)

    if condition and condition.strip():
        mask, _ = _build_3d_mask(X, Y, condition)
        Z_show = np.where(mask, Z, np.nan) if mask is not None else Z
    else:
        Z_show = Z

    fig = go.Figure(data=[
        go.Heatmap(
            x=x, y=y, z=Z_show,
            colorscale=colorscale.lower(),
            colorbar=dict(title='f(x,y)', x=1.02),
            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>f(x,y)=%{z:.4f}<extra></extra>',
        )
    ])
    fig.update_layout(
        title=dict(text=f'Heatmap: f(x,y) = {expr}', x=0.5),
        xaxis_title='x', yaxis_title='y',
        height=600, template='plotly_white',
    )
    return _fig_to_json(fig)


def generate_3d_wireframe(expr, x_min, x_max, y_min, y_max, resolution,
                          condition='', colorscale='Viridis'):
    """Generate a 3D wireframe plot of f(x, y)."""
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _eval_2d_function(expr, X, Y)
    Z = np.nan_to_num(Z, nan=0.0, posinf=1e10, neginf=-1e10)

    if condition and condition.strip():
        mask, _ = _build_3d_mask(X, Y, condition)
        Z_show = np.where(mask, Z, np.nan) if mask is not None else Z
    else:
        Z_show = Z

    fig = go.Figure(data=[
        go.Surface(
            x=X, y=Y, z=Z_show,
            colorscale=colorscale.lower(),
            opacity=0.6,
            contours={
                'x': dict(show=True, color='cyan', highlightcolor='cyan', project=dict(x=True)),
                'y': dict(show=True, color='magenta', highlightcolor='magenta', project=dict(y=True)),
                'z': dict(show=True, color='limegreen', highlightcolor='limegreen', project=dict(z=True)),
            },
            showscale=True,
            colorbar=dict(title='f(x,y)', x=1.02),
            hovertemplate='x=%{x:.3f}<br>y=%{y:.3f}<br>f(x,y)=%{z:.4f}<extra></extra>',
        )
    ])
    fig.update_layout(
        title=dict(text=f'3D Wireframe: f(x,y) = {expr}', x=0.5,
                   font=dict(size=16)),
        scene=dict(
            xaxis_title='<b>x</b>',
            yaxis_title='<b>y</b>',
            zaxis_title='<b>f(x,y)</b>',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3)),
            aspectmode='cube',
        ),
        height=650, margin=dict(l=0, r=0, t=60, b=0), template='plotly_white',
    )
    return _fig_to_json(fig)


def compute_3d_probability(expr, x_min, x_max, y_min, y_max, resolution,
                           condition=''):
    """
    Compute region probability: ∫∫_region f(x,y) dxdy / ∫∫_domain f(x,y) dxdy.
    Returns a dict with stats.
    """
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _eval_2d_function(expr, X, Y)
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    # Approximate integral via Riemann sum
    dx = (x_max - x_min) / (resolution - 1)
    dy = (y_max - y_min) / (resolution - 1)

    total_mass = float(np.sum(np.maximum(Z, 0.0)) * dx * dy)
    region_size = 0
    region_mass = 0.0
    mask_error = None

    if condition and condition.strip():
        mask, err = _build_3d_mask(X, Y, condition)
        if mask is None:
            mask_error = err
            mask = np.ones_like(X, dtype=bool)
        region_size = int(np.sum(mask))
        region_mass = float(np.sum(np.maximum(np.where(mask, Z, 0.0), 0.0)) * dx * dy)
    else:
        region_size = int(X.size)

    probability = region_mass / total_mass if total_mass > 0 else 0.0
    region_pct = region_size / X.size * 100 if X.size > 0 else 0.0

    return {
        'total_mass': total_mass,
        'region_mass': region_mass,
        'probability': probability,
        'region_pct': region_pct,
        'region_size': region_size,
        'total_size': int(X.size),
        'mask_error': mask_error,
    }


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------

def home(request):
    """Home page with hero section and feature cards."""
    return render(request, 'visualizer/home.html', {
        'title': 'Home',
        'dist_count': len(se.DISTRIBUTIONS),
    })


def plotting(request):
    """Main interactive plotting page."""
    dist_list = [{'slug': k, 'name': v['name'], 'type': v['type']}
                 for k, v in se.DISTRIBUTIONS.items()]
    return render(request, 'visualizer/plotting.html', {
        'title': 'Plotting',
        'distributions': dist_list,
        'distributions_json': json.dumps(dist_list),
    })


def distributions(request):
    """Gallery page showing all distributions with filter/search."""
    dist_type = request.GET.get('type', 'all')
    search = request.GET.get('search', '').strip().lower()

    items = []
    for slug, info in se.DISTRIBUTIONS.items():
        if dist_type != 'all' and info['type'] != dist_type:
            continue
        if search and search not in info['name'].lower() and search not in slug:
            continue
        items.append({
            'slug': slug,
            'name': info['name'],
            'type': info['type'],
            'description': info['description'],
            'formula': info['formula'],
            'params': info['params'],
            'applications': info['applications'],
        })

    return render(request, 'visualizer/distributions.html', {
        'title': 'Distributions',
        'distributions': items,
        'current_type': dist_type,
        'search': search,
    })


def distribution_detail(request, slug):
    """Detail page for a single distribution with plots."""
    info = se.DISTRIBUTIONS.get(slug)
    if info is None:
        return render(request, 'visualizer/404.html', status=404)

    # Generate default plots as JSON specs
    defaults = {p['name']: p['default'] for p in info['params']}
    pdf_spec, cdf_spec = generate_distribution_plots(slug, **defaults)

    return render(request, 'visualizer/distribution_detail.html', {
        'title': info['name'],
        'dist': info,
        'slug': slug,
        'pdf_spec_json': json.dumps(pdf_spec),
        'cdf_spec_json': json.dumps(cdf_spec),
        'defaults_json': json.dumps(defaults),
    })


def calculations(request):
    """Calculations page: descriptive stats + correlation + probability."""
    return render(request, 'visualizer/calculations.html', {
        'title': 'Calculations',
        'distributions': [{'slug': k, 'name': v['name']} for k, v in se.DISTRIBUTIONS.items()],
    })


def about(request):
    """About page."""
    return render(request, 'visualizer/about.html', {
        'title': 'About',
        'dist_count': len(se.DISTRIBUTIONS),
    })


def plotting_3d(request):
    """3D plotting page for f(x, y) expressions."""
    return render(request, 'visualizer/plotting_3d.html', {
        'title': '3D Plotting',
    })


# ---------------------------------------------------------------------------
# API views
# ---------------------------------------------------------------------------

@csrf_exempt
@require_POST
def api_generate_plot(request):
    """Generate plots via AJAX."""
    try:
        data = json.loads(request.body)
        plot_type = data.get('plot_type', 'pdf_cdf')

        result = {'success': True}

        # --- Custom function plotting ---
        if plot_type == 'custom_function':
            expr = data.get('expression', '')
            x_min = float(data.get('x_min', -5))
            x_max = float(data.get('x_max', 5))
            n_points = int(data.get('n_points', 500))
            func_name = data.get('func_name', 'f(x)')

            x_vals = np.linspace(x_min, x_max, n_points)
            # Evaluate the expression safely
            # User can use: x, np, math, pi, e, sin, cos, tan, log, sqrt, abs, exp
            import math
            safe_dict = {
                'x': x_vals, 'np': np, 'math': math,
                'pi': math.pi, 'e': math.e,
                'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                'log': np.log, 'log10': np.log10,
                'sqrt': np.sqrt, 'abs': np.abs, 'exp': np.exp,
                'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh,
                'arcsin': np.arcsin, 'arccos': np.arccos, 'arctan': np.arctan,
                'floor': np.floor, 'ceil': np.ceil, 'power': np.power,
                'erf': None, 'erfc': None,
            }
            try:
                from scipy.special import erf, erfc
                safe_dict['erf'] = erf
                safe_dict['erfc'] = erfc
            except ImportError:
                pass

            # Compile and evaluate
            code = compile(expr.strip(), '<string>', 'eval')
            y_vals = eval(code, {"__builtins__": {}}, safe_dict)
            y_vals = np.asarray(y_vals, dtype=float)

            fig = go.Figure(
                data=[go.Scatter(x=x_vals, y=y_vals, mode='lines',
                                 line=dict(color=COLORS['primary'], width=3),
                                 name=func_name,
                                 hovertemplate=f'x=%{{x:.4f}}<br>{func_name}=%{{y:.4f}}<extra></extra>')],
                layout=_make_plotly_layout(f'Custom Function: {func_name}',
                                           xlabel='x', ylabel=func_name)
            )

            # Area fill option
            fill_from = data.get('fill_from')
            fill_to = data.get('fill_to')
            if fill_from and fill_to:
                fill_from, fill_to = float(fill_from), float(fill_to)
                mask = (x_vals >= fill_from) & (x_vals <= fill_to)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[fill_from], x_vals[mask], [fill_to]]),
                    y=np.concatenate([[0], y_vals[mask], [0]]),
                    fill='tozeroy', fillcolor='rgba(64,224,208,0.3)',
                    line=dict(color='rgba(64,224,208,0)'),
                    showlegend=False, hoverinfo='skip',
                ))

            result['pdf'] = _fig_to_json(fig)
            return JsonResponse(result)

        # --- Standard distribution plotting ---
        slug = data.get('distribution')
        params = data.get('params', {})

        if not slug:
            return JsonResponse({'success': False, 'error': 'No distribution specified'}, status=400)

        if plot_type == 'side_by_side':
            slug2 = data.get('distribution2')
            params2 = data.get('params2', {})
            if slug2 and slug2 != slug:
                result['comparison'] = generate_side_by_side(slug, slug2, params, params2)
                return JsonResponse(result)

        if plot_type == 'joint':
            n = int(data.get('samples', 1000))
            result['joint'] = generate_joint_plot(slug, n_samples=n, **params)
        else:
            pdf_html, cdf_html = generate_distribution_plots(slug, **params)
            result['pdf'] = pdf_html
            result['cdf'] = cdf_html

        # Probability info
        try:
            prob_info = se.compute_probability(slug, **params)
            result['prob_info'] = prob_info
        except Exception:
            pass

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_POST
def api_calculate_stats(request):
    """Compute statistics from uploaded data."""
    try:
        data = json.loads(request.body) if request.body else {}
        values_text = data.get('values', '')
        x_values = data.get('x_values', '')
        y_values = data.get('y_values', '')
        corr_method = data.get('corr_method', 'both')
        prob_slug = data.get('prob_slug')
        prob_params = data.get('prob_params', {})
        prob_x = data.get('prob_x')
        prob_lower = data.get('prob_lower')
        prob_upper = data.get('prob_upper')

        result = {'success': True}

        # Descriptive stats from comma/space/newline separated values
        if values_text.strip():
            arr = [float(v.strip()) for v in re.split(r'[,\s\n]+', values_text.strip()) if v.strip()]
            if len(arr) > 1:
                result['descriptive'] = se.descriptive_stats(arr)
                result['data_preview'] = arr[:10]
                result['data_count'] = len(arr)

        # Correlation from paired data
        if x_values.strip() and y_values.strip():
            x_arr = [float(v.strip()) for v in re.split(r'[,\s\n]+', x_values.strip()) if v.strip()]
            y_arr = [float(v.strip()) for v in re.split(r'[,\s\n]+', y_values.strip()) if v.strip()]
            if len(x_arr) == len(y_arr) and len(x_arr) > 1:
                result['correlation'] = se.correlation_stats(x_arr, y_arr, method=corr_method)

        # Probability calculation
        if prob_slug:
            prob_params_clean = {k: float(v) for k, v in prob_params.items() if v.strip()}
            prob_result = se.compute_probability(
                prob_slug,
                x_value=float(prob_x) if prob_x else None,
                lower=float(prob_lower) if prob_lower else None,
                upper=float(prob_upper) if prob_upper else None,
                **prob_params_clean
            )
            result['probability'] = prob_result

        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_POST
def api_export_data(request):
    """Export generated data as CSV/JSON."""
    try:
        data = json.loads(request.body)
        slug = data.get('distribution')
        fmt = data.get('format', 'csv')
        n_samples = int(data.get('samples', 100))
        params = data.get('params', {})

        samples = se.sample(slug, n_samples=n_samples, **params)
        name = se.DISTRIBUTIONS.get(slug, {}).get('name', slug)

        if fmt == 'json':
            output = json.dumps({
                'distribution': name,
                'params': params,
                'n_samples': n_samples,
                'samples': samples.tolist(),
            })
            mime = 'application/json'
            ext = 'json'
        else:
            df = pd.DataFrame({'samples': samples})
            output = df.to_csv(index=False)
            mime = 'text/csv'
            ext = 'csv'

        encoded = base64.b64encode(output.encode()).decode()
        return JsonResponse({
            'success': True,
            'data': encoded,
            'mime': mime,
            'ext': ext,
            'filename': f'{slug}_samples.{ext}',
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)


@csrf_exempt
@require_POST
def api_generate_3d_plot(request):
    """Generate a 3D plot of f(x, y) via AJAX."""
    try:
        data = json.loads(request.body)
        expr = (data.get('expression') or '').strip()
        if not expr:
            return JsonResponse({'success': False, 'error': 'No expression provided'}, status=400)

        x_min = float(data.get('x_min', -3))
        x_max = float(data.get('x_max', 3))
        y_min = float(data.get('y_min', -3))
        y_max = float(data.get('y_max', 3))
        resolution = int(data.get('resolution', 60))
        resolution = max(10, min(200, resolution))
        plot_type = data.get('plot_type', 'surface')
        colorscale = data.get('colorscale', 'Viridis')
        opacity = float(data.get('opacity', 0.9))
        condition = (data.get('condition') or '').strip()

        if x_min >= x_max or y_min >= y_max:
            return JsonResponse({'success': False,
                                 'error': 'Min must be less than max for both axes'}, status=400)

        # Generate the requested plot
        if plot_type == 'surface':
            fig = generate_3d_surface(expr, x_min, x_max, y_min, y_max, resolution,
                                     condition=condition, colorscale=colorscale, opacity=opacity)
        elif plot_type == 'contour':
            fig = generate_3d_contour(expr, x_min, x_max, y_min, y_max, resolution,
                                      condition=condition, colorscale=colorscale)
        elif plot_type == 'heatmap':
            fig = generate_3d_heatmap(expr, x_min, x_max, y_min, y_max, resolution,
                                      condition=condition, colorscale=colorscale)
        elif plot_type == 'wireframe':
            fig = generate_3d_wireframe(expr, x_min, x_max, y_min, y_max, resolution,
                                        condition=condition, colorscale=colorscale)
        else:
            return JsonResponse({'success': False,
                                 'error': f'Unknown plot_type: {plot_type}'}, status=400)

        # Compute region probability
        prob = compute_3d_probability(expr, x_min, x_max, y_min, y_max, resolution,
                                      condition=condition)

        return JsonResponse({
            'success': True,
            'figure': fig,
            'plot_type': plot_type,
            'stats': prob,
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=400)
