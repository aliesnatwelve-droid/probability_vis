import sys, os
sys.path.insert(0, r'C:\Users\user\Desktop\probability_visualizer')
os.environ['DJANGO_SETTINGS_MODULE'] = 'config.settings'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Force fresh import by removing any cached modules
for mod in list(sys.modules.keys()):
    if 'visualizer' in mod or 'stats_engine' in mod or 'views' in mod:
        del sys.modules[mod]

import django
django.setup()

from visualizer import stats_engine as se

# Test
dist, args = se._build_params('normal', mu=0, sigma=1)
print(f'Normal: {args}')

# Generate plots
from visualizer.views import generate_distribution_plots
pdf, cdf = generate_distribution_plots('normal', mu=0, sigma=1)
print(f'PDF generated: {len(pdf)} chars')
print(f'CDF generated: {len(cdf)} chars')
print('OK')
